#include "wav2vec2.h"
#include "gguf.h"
#include "ops.h"
#include "ggml_backend.h"

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
    std::vector<float> weight; // [Cout, Cin, K]  — always F32 (conv is not a bottleneck)
    std::vector<float> bias;   // [Cout]
    std::vector<float> ln_w, ln_b;
    int Cout, Cin, K, stride;
    bool has_norm;
};

struct AttnLayer {
    ggml_tensor * q_w = nullptr; // [hidden, hidden]  quantized
    ggml_tensor * k_w = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * out_w = nullptr;
    std::vector<float> q_b, k_b, v_b, out_b; // biases stay F32
};

struct FFLayer {
    ggml_tensor * fc1_w = nullptr; // [intermediate, hidden]
    ggml_tensor * fc2_w = nullptr; // [hidden, intermediate]
    std::vector<float> fc1_b, fc2_b;
};

struct TransformerLayer {
    AttnLayer attn;
    FFLayer   ff;
    std::vector<float> ln1_w, ln1_b;
    std::vector<float> ln2_w, ln2_b;
};

struct Wav2Vec2Model {
    // hyperparams
    int n_conv_layers;
    int conv_dim;
    int n_enc_layers;
    int n_heads;
    int hidden;
    int intermediate;
    int vocab_size;
    int pad_id;
    std::string feat_norm;
    bool stable_layer_norm = false; // true = pre-norm (XLS-R), false = post-norm (base/large)
    std::vector<int> conv_kernel, conv_stride;
    int pos_conv_kernel, pos_conv_groups;

    // CNN feature extractor (small; kept as F32 vectors)
    std::vector<ConvLayer> conv_layers;

    // Feature projection: LN weights (conv_dim, F32) + projection (ggml, may be quantized)
    std::vector<float> proj_ln_w, proj_ln_b; // [conv_dim]
    ggml_tensor * proj_w = nullptr;           // [hidden, conv_dim]
    std::vector<float> proj_b;               // [hidden]

    // Positional conv (small; F32 vector)
    std::vector<float> pos_w, pos_b;

    // Encoder layer norm
    std::vector<float> enc_ln_w, enc_ln_b;

    // Transformer
    std::vector<TransformerLayer> enc_layers;

    // CTC head (ggml)
    ggml_tensor * lm_w = nullptr; // [vocab_size, hidden]
    std::vector<float> lm_b;     // [vocab_size]

    // Vocabulary
    std::vector<std::string> vocab;
    std::string word_delimiter;

    // ggml weight context — owns all ggml_tensor * above
    GgmlWeights gwts;
};

struct wav2vec2_context {
    Wav2Vec2Model model;
    GgmlScratch   scratch; // reused per-call compute scratch
};

// ════════════════════════════════════════════════════════════════════════════
// Model loading
// ════════════════════════════════════════════════════════════════════════════

// Return the GGML type enum (as uint32) for a loaded tensor, defaulting to F32.
static ggml_type tensor_ggml_type(const GgufTensor & gt) {
    return (ggml_type)gt.gtype; // GgufTensorType values match ggml_type values
}

// Compute total bytes needed in the ggml weight context for a set of tensors.
static size_t weight_ctx_size(const std::vector<const GgufTensor *> & tensors) {
    size_t total = 0;
    for (auto * gt : tensors) total += gt->raw.size();
    total += tensors.size() * ggml_tensor_overhead();
    return total;
}

// Add a 2D weight tensor to gwts from a GgufTensor.
// The weight matrix in HuggingFace convention is [out_features, in_features].
// ggml stores it as ne[0]=in_features (K), ne[1]=out_features (N) — same layout.
static ggml_tensor * add_weight(GgmlWeights & gwts, const GgufTensor & gt) {
    // shape in GGUF: innermost first, so shape[0]=in_features, shape[1]=out_features
    // This already matches ggml convention (ne[0]=K, ne[1]=N).
    int64_t ne0 = (int64_t)gt.shape[0];
    int64_t ne1 = gt.shape.size() > 1 ? (int64_t)gt.shape[1] : 1;
    return gwts.add(tensor_ggml_type(gt), ne0, ne1, gt.raw.data(), gt.raw.size());
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

    m.vocab          = gguf_arr_str(gf, "tokenizer.ggml.tokens");
    m.vocab_size     = (int)m.vocab.size();
    m.word_delimiter = gf.kv.count("wav2vec2.word_delimiter_token")
                     ? gf.kv.at("wav2vec2.word_delimiter_token").str
                     : "|";
    m.stable_layer_norm = gf.kv.count("wav2vec2.stable_layer_norm")
                        && gf.kv.at("wav2vec2.stable_layer_norm").u64 != 0;

    // ── Collect all linear tensors to size the ggml context ─────────────────
    std::vector<const GgufTensor *> linear_tensors;
    auto register_linear = [&](const std::string & name) {
        auto it = gf.tensors.find(name);
        if (it == gf.tensors.end())
            throw std::runtime_error("Missing tensor: " + name);
        linear_tensors.push_back(&it->second);
    };

    register_linear("feature_projection.projection.weight");
    for (int i = 0; i < m.n_enc_layers; i++) {
        std::string p = "encoder.layers." + std::to_string(i);
        register_linear(p + ".attention.q_proj.weight");
        register_linear(p + ".attention.k_proj.weight");
        register_linear(p + ".attention.v_proj.weight");
        register_linear(p + ".attention.out_proj.weight");
        register_linear(p + ".feed_forward.intermediate_dense.weight");
        register_linear(p + ".feed_forward.output_dense.weight");
    }
    register_linear("lm_head.weight");

    m.gwts.init(weight_ctx_size(linear_tensors));

    // ── CNN feature extractor (F32 vectors, not ggml) ────────────────────────
    m.conv_layers.resize(m.n_conv_layers);
    int in_ch = 1;
    for (int i = 0; i < m.n_conv_layers; i++) {
        auto & cl  = m.conv_layers[i];
        std::string p = "feature_extractor.conv_layers." + std::to_string(i);
        cl.weight = std::vector<float>(gf.tensors.at(p + ".conv.weight").data);
        if (gf.tensors.count(p + ".conv.bias"))
            cl.bias = std::vector<float>(gf.tensors.at(p + ".conv.bias").data);
        cl.Cout   = m.conv_dim;
        cl.Cin    = in_ch;
        cl.K      = m.conv_kernel[i];
        cl.stride = m.conv_stride[i];

        bool has = gf.tensors.count(p + ".layer_norm.weight") > 0;
        cl.has_norm = has;
        if (has) {
            cl.ln_w = gf.tensors.at(p + ".layer_norm.weight").data;
            cl.ln_b = gf.tensors.at(p + ".layer_norm.bias").data;
        }
        in_ch = m.conv_dim;
    }

    // ── Feature projection ───────────────────────────────────────────────────
    m.proj_ln_w = gf.tensors.at("feature_projection.layer_norm.weight").data;
    m.proj_ln_b = gf.tensors.at("feature_projection.layer_norm.bias").data;
    m.proj_b    = gf.tensors.at("feature_projection.projection.bias").data;
    m.proj_w    = add_weight(m.gwts, gf.tensors.at("feature_projection.projection.weight"));

    // ── Positional conv (F32 vector) ─────────────────────────────────────────
    m.pos_w = gf.tensors.at("encoder.pos_conv_embed.conv.weight").data;
    m.pos_b = gf.tensors.at("encoder.pos_conv_embed.conv.bias").data;

    // ── Encoder layer norm ───────────────────────────────────────────────────
    m.enc_ln_w = gf.tensors.at("encoder.layer_norm.weight").data;
    m.enc_ln_b = gf.tensors.at("encoder.layer_norm.bias").data;

    // ── Transformer layers ───────────────────────────────────────────────────
    m.enc_layers.resize(m.n_enc_layers);
    for (int i = 0; i < m.n_enc_layers; i++) {
        auto & el = m.enc_layers[i];
        std::string p = "encoder.layers." + std::to_string(i);

        el.attn.q_b   = gf.tensors.at(p + ".attention.q_proj.bias").data;
        el.attn.k_b   = gf.tensors.at(p + ".attention.k_proj.bias").data;
        el.attn.v_b   = gf.tensors.at(p + ".attention.v_proj.bias").data;
        el.attn.out_b = gf.tensors.at(p + ".attention.out_proj.bias").data;
        el.attn.q_w   = add_weight(m.gwts, gf.tensors.at(p + ".attention.q_proj.weight"));
        el.attn.k_w   = add_weight(m.gwts, gf.tensors.at(p + ".attention.k_proj.weight"));
        el.attn.v_w   = add_weight(m.gwts, gf.tensors.at(p + ".attention.v_proj.weight"));
        el.attn.out_w = add_weight(m.gwts, gf.tensors.at(p + ".attention.out_proj.weight"));

        el.ff.fc1_b   = gf.tensors.at(p + ".feed_forward.intermediate_dense.bias").data;
        el.ff.fc2_b   = gf.tensors.at(p + ".feed_forward.output_dense.bias").data;
        el.ff.fc1_w   = add_weight(m.gwts, gf.tensors.at(p + ".feed_forward.intermediate_dense.weight"));
        el.ff.fc2_w   = add_weight(m.gwts, gf.tensors.at(p + ".feed_forward.output_dense.weight"));

        el.ln1_w = gf.tensors.at(p + ".layer_norm.weight").data;
        el.ln1_b = gf.tensors.at(p + ".layer_norm.bias").data;
        el.ln2_w = gf.tensors.at(p + ".final_layer_norm.weight").data;
        el.ln2_b = gf.tensors.at(p + ".final_layer_norm.bias").data;
    }

    // ── CTC head ─────────────────────────────────────────────────────────────
    m.lm_b = gf.tensors.at("lm_head.bias").data;
    m.lm_w = add_weight(m.gwts, gf.tensors.at("lm_head.weight"));

    return ctx;
}

void wav2vec2_free(wav2vec2_context * ctx) { delete ctx; }

// ════════════════════════════════════════════════════════════════════════════
// Forward pass
// ════════════════════════════════════════════════════════════════════════════

static std::vector<float> feature_extract(const Wav2Vec2Model & m,
                                           const float * audio, int n) {
    std::vector<float> cur(audio, audio + n);
    int L = n;
    for (int i = 0; i < m.n_conv_layers; i++) {
        const auto & cl = m.conv_layers[i];
        auto next = wv2_conv1d(cur.data(), cl.weight.data(),
                               cl.bias.empty() ? nullptr : cl.bias.data(),
                               L, cl.Cin, cl.Cout, cl.K, cl.stride, 0);
        L = (L - cl.K) / cl.stride + 1;
        if (cl.has_norm) {
            if (m.feat_norm == "group")
                wv2_group_norm(next.data(), cl.ln_w.data(), cl.ln_b.data(), L, cl.Cout);
            else
                wv2_layer_norm(next.data(), cl.ln_w.data(), cl.ln_b.data(), L, cl.Cout);
        }
        for (auto & v : next) v = wv2_gelu(v);
        cur = std::move(next);
    }
    return cur; // [L, conv_dim]
}

// Multi-head self-attention with parallel heads.
// Q/K/V projections use ggml (dequantize-on-the-fly for quantized weights).
static void self_attention(const TransformerLayer & el, GgmlScratch & scratch,
                            float * x_out, const float * x_in,
                            int T, int H, int n_heads, int n_threads) {
    int d = H / n_heads;
    float scale = 1.f / sqrtf((float)d);

    // Q/K/V projections — uses ggml_mul_mat (handles quantized weights)
    auto Q = scratch.linear(el.attn.q_w, x_in, T, H, H, el.attn.q_b, n_threads);
    auto K = scratch.linear(el.attn.k_w, x_in, T, H, H, el.attn.k_b, n_threads);
    auto V = scratch.linear(el.attn.v_w, x_in, T, H, H, el.attn.v_b, n_threads);

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
                        out[i * H + h * d + k] = s;
                    }
            }
        });
    }
    for (auto & th : pool) th.join();

    auto proj = scratch.linear(el.attn.out_w, out.data(), T, H, H, el.attn.out_b, n_threads);
    memcpy(x_out, proj.data(), (size_t)T * H * sizeof(float));
}

static std::vector<float> encode(const Wav2Vec2Model & m, GgmlScratch & scratch,
                                  const float * feat, int T, int n_threads) {
    int H = m.hidden;

    // Feature projection: LN(conv_dim) → Linear(conv_dim → H)
    std::vector<float> feat_norm(feat, feat + T * m.conv_dim);
    wv2_layer_norm(feat_norm.data(), m.proj_ln_w.data(), m.proj_ln_b.data(), T, m.conv_dim);
    auto x = scratch.linear(m.proj_w, feat_norm.data(), T, m.conv_dim, H, m.proj_b, n_threads);

    // Positional conv
    int pk = m.pos_conv_kernel, pg = m.pos_conv_groups;
    auto pos = wv2_conv1d(x.data(), m.pos_w.data(), m.pos_b.data(),
                          T, H, H, pk, 1, pk / 2, pg);
    int use_T = std::min(T, (int)pos.size() / H);
    for (auto & v : pos) v = wv2_gelu(v);
    for (int t = 0; t < use_T; t++)
        for (int i = 0; i < H; i++) x[t * H + i] += pos[t * H + i];

    std::vector<float> attn_buf(T * H), ln_buf(T * H);

    if (!m.stable_layer_norm) {
        // Post-norm (base / large-960h): global LN before layers, per-layer LN after residual
        wv2_layer_norm(x.data(), m.enc_ln_w.data(), m.enc_ln_b.data(), T, H);
        for (const auto & el : m.enc_layers) {
            self_attention(el, scratch, attn_buf.data(), x.data(), T, H, m.n_heads, n_threads);
            for (int i = 0; i < T * H; i++) x[i] += attn_buf[i];
            wv2_layer_norm(x.data(), el.ln1_w.data(), el.ln1_b.data(), T, H);

            auto ff1 = scratch.linear(el.ff.fc1_w, x.data(), T, H, m.intermediate, el.ff.fc1_b, n_threads);
            for (auto & v : ff1) v = wv2_gelu(v);
            auto ff2 = scratch.linear(el.ff.fc2_w, ff1.data(), T, m.intermediate, H, el.ff.fc2_b, n_threads);
            for (int i = 0; i < T * H; i++) x[i] += ff2[i];
            wv2_layer_norm(x.data(), el.ln2_w.data(), el.ln2_b.data(), T, H);
        }
    } else {
        // Pre-norm / stable_layer_norm (XLS-R): per-layer LN before sublayer, global LN after all layers
        for (const auto & el : m.enc_layers) {
            // LN → attention → residual
            std::copy(x.begin(), x.end(), ln_buf.begin());
            wv2_layer_norm(ln_buf.data(), el.ln1_w.data(), el.ln1_b.data(), T, H);
            self_attention(el, scratch, attn_buf.data(), ln_buf.data(), T, H, m.n_heads, n_threads);
            for (int i = 0; i < T * H; i++) x[i] += attn_buf[i];

            // LN → feedforward → residual
            std::copy(x.begin(), x.end(), ln_buf.begin());
            wv2_layer_norm(ln_buf.data(), el.ln2_w.data(), el.ln2_b.data(), T, H);
            auto ff1 = scratch.linear(el.ff.fc1_w, ln_buf.data(), T, H, m.intermediate, el.ff.fc1_b, n_threads);
            for (auto & v : ff1) v = wv2_gelu(v);
            auto ff2 = scratch.linear(el.ff.fc2_w, ff1.data(), T, m.intermediate, H, el.ff.fc2_b, n_threads);
            for (int i = 0; i < T * H; i++) x[i] += ff2[i];
        }
        wv2_layer_norm(x.data(), m.enc_ln_w.data(), m.enc_ln_b.data(), T, H);
    }
    return x;
}

static bool is_special_token(const std::string & s) {
    return s.size() >= 2 && s.front() == '<' && s.back() == '>';
}

static std::string ids_to_text(const std::vector<int> & ids, int blank_id,
                                const std::vector<std::string> & vocab,
                                const std::string & word_delim) {
    std::string result;
    for (int id : ids) {
        if (id == blank_id || id < 0 || id >= (int)vocab.size()) continue;
        const auto & tok = vocab[id];
        if (tok == word_delim)      { result += ' '; continue; }
        if (is_special_token(tok))  continue;
        result += tok;
    }
    auto start = result.find_first_not_of(' ');
    auto end   = result.find_last_not_of(' ');
    if (start == std::string::npos) return "";
    return result.substr(start, end - start + 1);
}

struct WordSpan { std::string word; int t_start; int t_end; };

// Convert per-frame spans → word-level spans with timing.
// Frame index → seconds: t * 320 / 16000 = t * 0.02
static std::vector<WordSpan> spans_to_words(
    const std::vector<Wv2TokenSpan> & spans,
    int blank_id, const std::vector<std::string> & vocab,
    const std::string & word_delim)
{
    std::vector<WordSpan> words;
    std::string cur_word;
    int cur_start = -1, cur_end = -1;

    auto flush = [&]() {
        if (cur_word.empty()) return;
        // trim leading/trailing spaces
        auto s = cur_word.find_first_not_of(' ');
        auto e = cur_word.find_last_not_of(' ');
        if (s != std::string::npos)
            words.push_back({cur_word.substr(s, e - s + 1), cur_start, cur_end});
        cur_word.clear(); cur_start = -1; cur_end = -1;
    };

    for (const auto & sp : spans) {
        int id = sp.token_id;
        if (id == blank_id || id < 0 || id >= (int)vocab.size()) continue;
        const auto & tok = vocab[id];
        if (is_special_token(tok)) continue;
        if (tok == word_delim) { flush(); continue; }
        if (cur_start < 0) cur_start = sp.t_start;
        cur_end  = sp.t_end;
        cur_word += tok;
    }
    flush();
    return words;
}

// ════════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════════

wav2vec2_params wav2vec2_default_params() {
    wav2vec2_params p{};
    p.n_threads       = 4;
    p.beam_size       = 1;
    p.verbose         = false;
    p.word_timestamps = false;
    return p;
}

void wav2vec2_result_free(wav2vec2_result * res) {
    if (!res) return;
    free(res->transcript);
    if (res->words) {
        for (int i = 0; i < res->n_words; i++) free(res->words[i].text);
        free(res->words);
    }
    free(res);
}

wav2vec2_result * wav2vec2_transcribe(wav2vec2_context * ctx,
                                       wav2vec2_params     params,
                                       const float       * samples,
                                       int                 n_samples) {
    const auto & m = ctx->model;

    auto t0 = std::chrono::steady_clock::now();

    auto feat = feature_extract(m, samples, n_samples);
    int T = (int)feat.size() / m.conv_dim;

    auto enc = encode(m, ctx->scratch, feat.data(), T, params.n_threads);
    auto logits = ctx->scratch.linear(m.lm_w, enc.data(), T, m.hidden, m.vocab_size, m.lm_b, params.n_threads);

    std::string text;
    std::vector<WordSpan> word_spans;

    if (params.beam_size > 1) {
        // Beam search: no per-frame alignment, timestamps not supported
        auto ids = wv2_ctc_beam_search(logits.data(), T, m.vocab_size, m.pad_id, params.beam_size);
        text = ids_to_text(ids, m.pad_id, m.vocab, m.word_delimiter);
    } else {
        auto spans = wv2_ctc_greedy_spans(logits.data(), T, m.vocab_size);
        std::vector<int> ids;
        for (const auto & sp : spans)
            if (sp.token_id != m.pad_id) ids.push_back(sp.token_id);
        text = ids_to_text(ids, m.pad_id, m.vocab, m.word_delimiter);
        if (params.word_timestamps)
            word_spans = spans_to_words(spans, m.pad_id, m.vocab, m.word_delimiter);
    }

    if (params.verbose) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "wav2vec2: %d frames, %s(b=%d), %lld ms\n",
                T, params.beam_size > 1 ? "beam" : "greedy",
                params.beam_size, (long long)ms);
    }

    auto * res       = (wav2vec2_result *)calloc(1, sizeof(wav2vec2_result));
    res->n_frames   = T;
    res->transcript = (char *)malloc(text.size() + 1);
    memcpy(res->transcript, text.c_str(), text.size() + 1);

    if (params.word_timestamps && !word_spans.empty()) {
        res->n_words = (int)word_spans.size();
        res->words   = (wav2vec2_word *)malloc(res->n_words * sizeof(wav2vec2_word));
        for (int i = 0; i < res->n_words; i++) {
            const auto & ws = word_spans[i];
            res->words[i].t_start = ws.t_start * (320.f / 16000.f);
            res->words[i].t_end   = ws.t_end   * (320.f / 16000.f);
            res->words[i].text    = (char *)malloc(ws.word.size() + 1);
            memcpy(res->words[i].text, ws.word.c_str(), ws.word.size() + 1);
        }
    }
    return res;
}
