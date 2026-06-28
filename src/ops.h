// wav2vec2.cpp — math primitives (all inline, no deps beyond <cmath> unless WV2_USE_BLAS)
#pragma once

#include <cmath>
#include <cstring>
#include <vector>
#include <map>
#include <algorithm>
#include <cassert>
#include <limits>

#ifdef WV2_USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif

static inline float wv2_gelu(float x) {
    const float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// Layer-norm over last dim: x [T, D] in-place
static inline void wv2_layer_norm(float * x, const float * w, const float * b,
                                   int T, int D, float eps = 1e-5f) {
    for (int t = 0; t < T; t++) {
        float * row = x + t * D;
        float mean = 0.f, var = 0.f;
        for (int i = 0; i < D; i++) mean += row[i];
        mean /= D;
        for (int i = 0; i < D; i++) { float d = row[i] - mean; var += d * d; }
        var  /= D;
        float inv = 1.f / sqrtf(var + eps);
        for (int i = 0; i < D; i++) row[i] = (row[i] - mean) * inv * w[i] + b[i];
    }
}

// GroupNorm(num_groups=C): normalise each channel over T (wav2vec2-base extractor)
static inline void wv2_group_norm(float * x, const float * w, const float * b,
                                   int T, int C, float eps = 1e-5f) {
    for (int c = 0; c < C; c++) {
        float mean = 0.f, var = 0.f;
        for (int t = 0; t < T; t++) mean += x[t * C + c];
        mean /= T;
        for (int t = 0; t < T; t++) { float d = x[t * C + c] - mean; var += d * d; }
        var  /= T;
        float inv = 1.f / sqrtf(var + eps);
        for (int t = 0; t < T; t++) x[t * C + c] = (x[t * C + c] - mean) * inv * w[c] + b[c];
    }
}

// y [M, N] = x [M, K] @ W^T [N, K] + b [N]
// Uses CBLAS sgemm when available, falls back to naive otherwise.
static inline std::vector<float> wv2_linear(const float * x, const std::vector<float> & W,
                                             const std::vector<float> & b, int M, int K, int N) {
    std::vector<float> y(M * N, 0.f);
#ifdef WV2_USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, x, K, W.data(), K, 0.0f, y.data(), N);
#else
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0.f;
            for (int k = 0; k < K; k++) s += x[m * K + k] * W[n * K + k];
            y[m * N + n] = s;
        }
#endif
    if (!b.empty())
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) y[m * N + n] += b[n];
    return y;
}

// Conv1D: x [L, Cin] → out [L_out, Cout]
// weight [Cout, Cin/groups, K], bias [Cout]
static inline std::vector<float> wv2_conv1d(const float * x,
                                             const float * w, const float * b,
                                             int L, int Cin, int Cout, int K,
                                             int stride, int padding, int groups = 1) {
    assert(Cin % groups == 0 && Cout % groups == 0);
    int Cin_g  = Cin  / groups;
    int Cout_g = Cout / groups;
    int L_out  = (L + 2 * padding - K) / stride + 1;
    std::vector<float> out(L_out * Cout, 0.f);
    for (int g = 0; g < groups; g++)
        for (int t = 0; t < L_out; t++)
            for (int oc = 0; oc < Cout_g; oc++) {
                float s = b ? b[g * Cout_g + oc] : 0.f;
                for (int ic = 0; ic < Cin_g; ic++)
                    for (int k = 0; k < K; k++) {
                        int src = t * stride + k - padding;
                        if (src >= 0 && src < L)
                            s += x[src * Cin + g * Cin_g + ic]
                               * w[(g * Cout_g + oc) * (Cin_g * K) + ic * K + k];
                    }
                out[t * Cout + g * Cout_g + oc] = s;
            }
    return out;
}

static inline void wv2_softmax(float * x, int n) {
    float mx = *std::max_element(x, x + n);
    float s = 0.f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

// Greedy CTC decode: logits [T, V] → collapsed token id sequence (includes blanks)
static inline std::vector<int> wv2_ctc_greedy(const float * logits, int T, int V) {
    std::vector<int> ids;
    int prev = -1;
    for (int t = 0; t < T; t++) {
        int best = 0;
        float best_v = logits[t * V];
        for (int v = 1; v < V; v++)
            if (logits[t * V + v] > best_v) { best_v = logits[t * V + v]; best = v; }
        if (best != prev) { ids.push_back(best); prev = best; }
    }
    return ids;
}

// Per-emission span for word-level timestamps.
// t_start / t_end are CTC frame indices (frame * 320 / 16000 = seconds).
struct Wv2TokenSpan {
    int token_id;
    int t_start;   // first frame emitting this token
    int t_end;     // one past last frame of this run
};

// Greedy CTC decode with frame-level timing.
// Returns one span per unique run after duplicate collapse (includes blank spans).
static inline std::vector<Wv2TokenSpan> wv2_ctc_greedy_spans(
    const float * logits, int T, int V)
{
    std::vector<Wv2TokenSpan> spans;
    int prev = -1, run_start = 0;
    for (int t = 0; t < T; t++) {
        int best = 0;
        float best_v = logits[t * V];
        for (int v = 1; v < V; v++)
            if (logits[t * V + v] > best_v) { best_v = logits[t * V + v]; best = v; }
        if (best != prev) {
            if (prev != -1) spans.push_back({prev, run_start, t});
            prev = best; run_start = t;
        }
    }
    if (prev != -1) spans.push_back({prev, run_start, T});
    return spans;
}

// ── CTC Beam Search ──────────────────────────────────────────────────────────
// Returns decoded token ids (no blanks, dups already collapsed).
// Implements Graves 2006 prefix beam search in log domain.

static inline float wv2_log_sum_exp(float a, float b) {
    if (a <= -1e29f) return b;
    if (b <= -1e29f) return a;
    float mx = std::max(a, b);
    return mx + log1pf(expf(std::min(a, b) - mx));
}

static inline std::vector<int> wv2_ctc_beam_search(
    const float * logits, int T, int V, int blank_id, int beam_width)
{
    static constexpr float kNegInf = -1e30f;

    struct BeamState {
        float log_p_b  = kNegInf;   // log prob of paths ending in blank
        float log_p_nb = kNegInf;   // log prob of paths ending in non-blank
        float score()  const { return wv2_log_sum_exp(log_p_b, log_p_nb); }
    };
    using BeamMap = std::map<std::vector<int>, BeamState>;

    BeamMap cur;
    cur[{}].log_p_b = 0.f;   // empty prefix, came from blank, P=1

    std::vector<float> log_probs(V);

    for (int t = 0; t < T; t++) {
        // log-softmax for numerical stability
        const float * row = logits + t * V;
        float mx = *std::max_element(row, row + V);
        float Z = 0.f;
        for (int v = 0; v < V; v++) Z += expf(row[v] - mx);
        float log_Z = mx + logf(Z);
        for (int v = 0; v < V; v++) log_probs[v] = row[v] - log_Z;

        BeamMap next;

        for (auto & [tokens, state] : cur) {
            int last = tokens.empty() ? -1 : tokens.back();

            // Extend with blank — prefix stays the same
            {
                float lp = log_probs[blank_id];
                auto & ns = next[tokens];
                ns.log_p_b = wv2_log_sum_exp(ns.log_p_b,
                    wv2_log_sum_exp(state.log_p_b, state.log_p_nb) + lp);
            }

            // Extend with each non-blank character
            for (int c = 0; c < V; c++) {
                if (c == blank_id) continue;
                float lp = log_probs[c];

                if (c == last) {
                    // Same char from non-blank: stays same prefix (consecutive run)
                    {
                        auto & ns = next[tokens];
                        ns.log_p_nb = wv2_log_sum_exp(ns.log_p_nb, state.log_p_nb + lp);
                    }
                    // Same char from blank: blank separates them → extends prefix
                    {
                        auto new_tok = tokens;
                        new_tok.push_back(c);
                        auto & ns = next[new_tok];
                        ns.log_p_nb = wv2_log_sum_exp(ns.log_p_nb, state.log_p_b + lp);
                    }
                } else {
                    // Different char: extends prefix
                    auto new_tok = tokens;
                    new_tok.push_back(c);
                    auto & ns = next[new_tok];
                    ns.log_p_nb = wv2_log_sum_exp(ns.log_p_nb,
                        wv2_log_sum_exp(state.log_p_b, state.log_p_nb) + lp);
                }
            }
        }

        // Prune to top beam_width by total probability
        cur.clear();
        std::vector<std::pair<float, const std::vector<int> *>> ranked;
        ranked.reserve(next.size());
        for (auto & [tok, st] : next)
            ranked.push_back({st.score(), &tok});
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto & a, const auto & b){ return a.first > b.first; });
        int keep = std::min((int)ranked.size(), beam_width);
        for (int i = 0; i < keep; i++)
            cur[*ranked[i].second] = next[*ranked[i].second];
    }

    // Return highest-scoring beam
    const std::vector<int> * best = nullptr;
    float best_score = kNegInf;
    for (auto & [tok, st] : cur) {
        float s = st.score();
        if (s > best_score) { best_score = s; best = &tok; }
    }
    return best ? *best : std::vector<int>{};
}
