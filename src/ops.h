// wav2vec2.cpp — math primitives (all inline, no deps beyond <cmath>)
#pragma once

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cassert>

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
static inline std::vector<float> wv2_linear(const float * x, const std::vector<float> & W,
                                             const std::vector<float> & b, int M, int K, int N) {
    std::vector<float> y(M * N);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = b.empty() ? 0.f : b[n];
            for (int k = 0; k < K; k++) s += x[m * K + k] * W[n * K + k];
            y[m * N + n] = s;
        }
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

// Greedy CTC decode: logits [T, V] → token id sequence
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
