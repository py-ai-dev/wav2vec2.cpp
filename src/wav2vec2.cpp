#include "wav2vec2.h"
#include "gguf.h"
#include "ops.h"

#include <cstring>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <thread>
#include <algorithm>

// ════════════════════════════════════════════════════════════════════════════
// Model structs
// ════════════════════════════════════════════════════════════════════════════

struct ConvLayer {
    std::vector<float> weight; // [Cout, Cin, K]
    std::vector<float> bias;   // [Cout]
    std::vector<float> ln_w, ln_b; // layer/group norm (may be empty)
    int Cout, Cin, K, stride;
    bool has_norm;
};

struct AttnLayer {
    std::vector<float> q_w, q_b, k_w, k_b, v_w, v_b;
    std::vector<float> out_w, out_b;
};

struct FFLayer {
    std::vector<float> fc1_w, fc1_b;
    std::vector<float> fc2_w, fc2_b;
};

struct TransformerLayer {
    AttnLayer attn;
    FFLayer   ff;
    std::vector<float> ln1_w, ln1_b;   // post-attention layer norm
    std::vector<float> ln2_w, ln2_b;   // post-FFN layer norm
};

struct Wav2Vec2Model {
    // hyperparams
    int n_conv_layers;
    int conv_dim;           // 512
    int n_enc_layers;       // 12/24
    int n_heads;            // 12/16
    int hidden;             // 768/1024
    int intermediate;       // 3072/4096
    int vocab_size;
    int pad_id;             // CTC blank
    std::string feat_norm;  // "group" | "layer"
    std::vector<int> conv_kernel, conv_stride;
    int pos_conv_kernel, pos_conv_groups;

    // weights
    std::vector<ConvLayer> conv_layers;

    // feature projection (HF order: LayerNorm(conv_dim) → Linear(conv_dim→hidden))
    std::vector<float> proj_ln_w, proj_ln_b; // [conv_dim]
    std::vector<float> proj_w, proj_b;        // proj_w: [hidden, conv_dim]

    // positional conv
    std::vector<float> pos_w, pos_b;

    // encoder layer norm
    std::vector<float> enc_ln_w, enc_ln_b;

    // transformer
    std::vector<TransformerLayer> enc_layers;

    // CTC head
    std::vector<float> lm_w, lm_b;

    // vocabulary: id → token string
    std::vector<std::string> vocab;
    std::string word_delimiter; // typically "|"
};

struct wav2vec2_context {
    Wav2Vec2Model model;
};

// ════════════════════════════════════════════════════════════════════════════
// Model loading
// ════════════════════════════════════════════════════════════════════════════

static std::vector<float> load_vec(const GgufFile & g, const std::string & name) {
    return std::vector<float>(g.tensors.at(name).data);
}

wav2vec2_context * wav2vec2_init(const char * path) {
    GgufFile gf = gguf_load(path);
    auto * ctx = new wav2vec2_context();
    auto & m   = ctx->model;

    // Hyperparams
    m.n_conv_layers   = gguf_u32(gf, "wav2vec2.n_conv_layers");
    m.conv_dim        = gguf_u32(gf, "wav2vec2.conv_dim");
    m.n_enc_layers    = gguf_u32(gf, "wav2vec2.n_encoder_layers");
    m.n_heads         = gguf_u32(gf, "wav2vec2.n_heads");
    m.hidden          = gguf_u32(gf, "wav2vec2.hidden_size");
    m.intermediate    = gguf_u32(gf, "wav2vec2.intermediate_size");
    m.vocab_size      = gguf_u32(gf, "wav2vec2.vocab_size");
    m.pad_id          = gguf_u32(gf, "wav2vec2.pad_token_id");
    m.feat_norm       = gguf_str(gf, "wav2vec2.feat_extract_norm");
    m.pos_conv_kernel = gguf_u32(gf, "wav2vec2.pos_conv_kernel");
    m.pos_conv_groups = gguf_u32(gf, "wav2vec2.pos_conv_groups");

    auto ck = gguf_arr_u32(gf, "wav2vec2.conv_kernel");
    auto cs = gguf_arr_u32(gf, "wav2vec2.conv_stride");
    for (auto v : ck) m.conv_kernel.push_back((int)v);
    for (auto v : cs) m.conv_stride.push_back((int)v);

    // Vocabulary
    m.vocab      = gguf_arr_str(gf, "tokenizer.ggml.tokens");
    m.vocab_size = (int)m.vocab.size();
    m.word_delimiter = "|";

    // CNN feature extractor
    m.conv_layers.resize(m.n_conv_layers);
    int in_ch = 1;
    for (int i = 0; i < m.n_conv_layers; i++) {
        auto & cl  = m.conv_layers[i];
        std::string p = "feature_extractor.conv_layers." + std::to_string(i);
        cl.weight = load_vec(gf, p + ".conv.weight");
        cl.bias   = load_vec(gf, p + ".conv.bias");
        cl.Cout   = m.conv_dim;
        cl.Cin    = in_ch;
        cl.K      = m.conv_kernel[i];
        cl.stride = m.conv_stride[i];

        bool has = gf.tensors.count(p + ".layer_norm.weight") > 0;
        cl.has_norm = has;
        if (has) {
            cl.ln_w = load_vec(gf, p + ".layer_norm.weight");
            cl.ln_b = load_vec(gf, p + ".layer_norm.bias");
        }
        in_ch = m.conv_dim;
    }

    // Feature projection: LN weights are [conv_dim], proj weights are [hidden, conv_dim]
    m.proj_ln_w = load_vec(gf, "feature_projection.layer_norm.weight");
    m.proj_ln_b = load_vec(gf, "feature_projection.layer_norm.bias");
    m.proj_w    = load_vec(gf, "feature_projection.projection.weight");
    m.proj_b    = load_vec(gf, "feature_projection.projection.bias");

    // Positional conv
    m.pos_w = load_vec(gf, "encoder.pos_conv_embed.conv.weight");
    m.pos_b = load_vec(gf, "encoder.pos_conv_embed.conv.bias");

    // Encoder layer norm
    m.enc_ln_w = load_vec(gf, "encoder.layer_norm.weight");
    m.enc_ln_b = load_vec(gf, "encoder.layer_norm.bias");

    // Transformer layers
    m.enc_layers.resize(m.n_enc_layers);
    for (int i = 0; i < m.n_enc_layers; i++) {
        auto & el = m.enc_layers[i];
        std::string p = "encoder.layers." + std::to_string(i);
        el.attn.q_w   = load_vec(gf, p + ".attention.q_proj.weight");
        el.attn.q_b   = load_vec(gf, p + ".attention.q_proj.bias");
        el.attn.k_w   = load_vec(gf, p + ".attention.k_proj.weight");
        el.attn.k_b   = load_vec(gf, p + ".attention.k_proj.bias");
        el.attn.v_w   = load_vec(gf, p + ".attention.v_proj.weight");
        el.attn.v_b   = load_vec(gf, p + ".attention.v_proj.bias");
        el.attn.out_w = load_vec(gf, p + ".attention.out_proj.weight");
        el.attn.out_b = load_vec(gf, p + ".attention.out_proj.bias");
        el.ff.fc1_w   = load_vec(gf, p + ".feed_forward.intermediate_dense.weight");
        el.ff.fc1_b   = load_vec(gf, p + ".feed_forward.intermediate_dense.bias");
        el.ff.fc2_w   = load_vec(gf, p + ".feed_forward.output_dense.weight");
        el.ff.fc2_b   = load_vec(gf, p + ".feed_forward.output_dense.bias");
        el.ln1_w      = load_vec(gf, p + ".layer_norm.weight");
        el.ln1_b      = load_vec(gf, p + ".layer_norm.bias");
        el.ln2_w      = load_vec(gf, p + ".final_layer_norm.weight");
        el.ln2_b      = load_vec(gf, p + ".final_layer_norm.bias");
    }

    // CTC head
    m.lm_w = load_vec(gf, "lm_head.weight");
    m.lm_b = load_vec(gf, "lm_head.bias");

    return ctx;
}

void wav2vec2_free(wav2vec2_context * ctx) { delete ctx; }

// ════════════════════════════════════════════════════════════════════════════
// Forward pass
// ════════════════════════════════════════════════════════════════════════════

static std::vector<float> feature_extract(const Wav2Vec2Model & m,
                                           const float * audio, int n) {
    std::vector<float> cur(audio, audio + n); // [n, 1] (Cin=1)

    int L = n;
    for (int i = 0; i < m.n_conv_layers; i++) {
        const auto & cl = m.conv_layers[i];
        auto next = wv2_conv1d(cur.data(), cl.weight.data(), cl.bias.data(),
                               L, cl.Cin, cl.Cout, cl.K, cl.stride, /*pad=*/0);
        L = (L - cl.K) / cl.stride + 1;

        // HF order: conv → norm → gelu
        if (cl.has_norm) {
            if (m.feat_norm == "group") {
                wv2_group_norm(next.data(), cl.ln_w.data(), cl.ln_b.data(), L, cl.Cout);
            } else {
                wv2_layer_norm(next.data(), cl.ln_w.data(), cl.ln_b.data(), L, cl.Cout);
            }
        }
        for (auto & v : next) v = wv2_gelu(v);

        cur = std::move(next);
    }
    return cur; // [L, conv_dim]
}

// Multi-head self-attention with parallel head computation.
// Heads are split across n_threads — writes to disjoint slices of H, no mutex needed.
static void self_attention(const TransformerLayer & el, float * x_out,
                            const float * x_in, int T, int H, int n_heads, int n_threads) {
    int d = H / n_heads;
    float scale = 1.f / sqrtf((float)d);

    // Q/K/V: [T, H]  — BLAS-accelerated when WV2_USE_BLAS is defined
    auto Q = wv2_linear(x_in, el.attn.q_w, el.attn.q_b, T, H, H);
    auto K = wv2_linear(x_in, el.attn.k_w, el.attn.k_b, T, H, H);
    auto V = wv2_linear(x_in, el.attn.v_w, el.attn.v_b, T, H, H);

    std::vector<float> out(T * H, 0.f);

    int n_t = std::min(n_threads, n_heads);
    std::vector<std::thread> pool;
    pool.reserve(n_t);

    for (int tid = 0; tid < n_t; tid++) {
        int h_lo = tid * n_heads / n_t;
        int h_hi = (tid + 1) * n_heads / n_t;
        pool.emplace_back([&, h_lo, h_hi]() {
            for (int h = h_lo; h < h_hi; h++) {
                std::vector<float> scores(T * T);
                for (int i = 0; i < T; i++)
                    for (int j = 0; j < T; j++) {
                        float s = 0.f;
                        for (int k = 0; k < d; k++)
                            s += Q[i * H + h * d + k] * K[j * H + h * d + k];
                        scores[i * T + j] = s * scale;
                    }
                for (int i = 0; i < T; i++) wv2_softmax(&scores[i * T], T);
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < d; k++) {
                        float s = 0.f;
                        for (int j = 0; j < T; j++)
                            s += scores[i * T + j] * V[j * H + h * d + k];
                        out[i * H + h * d + k] = s;  // disjoint slice per head
                    }
            }
        });
    }
    for (auto & th : pool) th.join();

    auto proj = wv2_linear(out.data(), el.attn.out_w, el.attn.out_b, T, H, H);
    memcpy(x_out, proj.data(), T * H * sizeof(float));
}

static std::vector<float> encode(const Wav2Vec2Model & m,
                                  const float * feat, int T, int n_threads) {
    int H = m.hidden;

    // Feature projection: LayerNorm(conv_dim) first, then Linear(conv_dim → hidden).
    // Matches HF's Wav2Vec2FeatureProjection: self.layer_norm → self.projection.
    std::vector<float> feat_norm(feat, feat + T * m.conv_dim);
    wv2_layer_norm(feat_norm.data(), m.proj_ln_w.data(), m.proj_ln_b.data(), T, m.conv_dim);
    auto x = wv2_linear(feat_norm.data(), m.proj_w, m.proj_b, T, m.conv_dim, H);

    // Positional conv embedding: grouped Conv1D + gelu, add to x
    int pk = m.pos_conv_kernel, pg = m.pos_conv_groups;
    auto pos = wv2_conv1d(x.data(), m.pos_w.data(), m.pos_b.data(),
                          T, H, H, pk, 1, /*pad=*/pk / 2, pg);
    int use_T = std::min(T, (int)pos.size() / H);
    for (auto & v : pos) v = wv2_gelu(v);
    for (int t = 0; t < use_T; t++)
        for (int i = 0; i < H; i++) x[t * H + i] += pos[t * H + i];

    // Pre-transformer layer norm
    wv2_layer_norm(x.data(), m.enc_ln_w.data(), m.enc_ln_b.data(), T, H);

    // Transformer layers
    std::vector<float> attn_buf(T * H);
    for (const auto & el : m.enc_layers) {
        // Self-attention with residual, then post-norm
        self_attention(el, attn_buf.data(), x.data(), T, H, m.n_heads, n_threads);
        for (int i = 0; i < T * H; i++) x[i] += attn_buf[i];
        wv2_layer_norm(x.data(), el.ln1_w.data(), el.ln1_b.data(), T, H);

        // Feed-forward with residual, then post-norm
        auto ff1 = wv2_linear(x.data(), el.ff.fc1_w, el.ff.fc1_b, T, H, m.intermediate);
        for (auto & v : ff1) v = wv2_gelu(v);
        auto ff2 = wv2_linear(ff1.data(), el.ff.fc2_w, el.ff.fc2_b, T, m.intermediate, H);
        for (int i = 0; i < T * H; i++) x[i] += ff2[i];
        wv2_layer_norm(x.data(), el.ln2_w.data(), el.ln2_b.data(), T, H);
    }
    return x; // [T, H]
}

// Map decoded token ids → transcript string
static std::string ids_to_text(const std::vector<int> & ids, int blank_id,
                                const std::vector<std::string> & vocab,
                                const std::string & word_delim) {
    std::string result;
    for (int id : ids) {
        if (id == blank_id || id < 0 || id >= (int)vocab.size()) continue;
        const auto & tok = vocab[id];
        result += (tok == word_delim) ? " " : tok;
    }
    auto start = result.find_first_not_of(' ');
    auto end   = result.find_last_not_of(' ');
    if (start == std::string::npos) return "";
    return result.substr(start, end - start + 1);
}

// ════════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════════

wav2vec2_params wav2vec2_default_params() {
    wav2vec2_params p{};
    p.n_threads = 4;
    p.beam_size = 1;
    p.verbose   = false;
    return p;
}

wav2vec2_result * wav2vec2_transcribe(wav2vec2_context * ctx,
                                       wav2vec2_params     params,
                                       const float       * samples,
                                       int                 n_samples) {
    const auto & m = ctx->model;

    auto t0 = std::chrono::steady_clock::now();

    // 1. Feature extraction
    auto feat = feature_extract(m, samples, n_samples);
    int T = (int)feat.size() / m.conv_dim;

    // 2. Encoder (attention parallelised over n_threads)
    auto enc = encode(m, feat.data(), T, params.n_threads);

    // 3. CTC head
    auto logits = wv2_linear(enc.data(), m.lm_w, m.lm_b, T, m.hidden, m.vocab_size);

    // 4. Decode — beam search or greedy
    std::string text;
    if (params.beam_size > 1) {
        auto ids = wv2_ctc_beam_search(logits.data(), T, m.vocab_size,
                                       m.pad_id, params.beam_size);
        text = ids_to_text(ids, m.pad_id, m.vocab, m.word_delimiter);
    } else {
        auto ids = wv2_ctc_greedy(logits.data(), T, m.vocab_size);
        std::vector<int> filtered;
        for (int id : ids)
            if (id != m.pad_id) filtered.push_back(id);
        text = ids_to_text(filtered, m.pad_id, m.vocab, m.word_delimiter);
    }

    if (params.verbose) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "wav2vec2: %d frames, %s(b=%d), %lld ms\n",
                T, params.beam_size > 1 ? "beam" : "greedy",
                params.beam_size, (long long)ms);
    }

    auto * res = (wav2vec2_result *)malloc(sizeof(wav2vec2_result));
    res->n_frames   = T;
    res->transcript = (char *)malloc(text.size() + 1);
    memcpy(res->transcript, text.c_str(), text.size() + 1);
    return res;
}
