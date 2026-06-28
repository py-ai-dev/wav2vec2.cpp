// ggml integration for wav2vec2.cpp
// Manages a persistent weight context (may hold quantized tensors) and provides
// a linear() call that uses ggml_mul_mat for dequantize-on-the-fly matmul.
#pragma once

#include "ggml.h"
#include "ggml-cpu.h"

#include <vector>
#include <cstring>
#include <stdexcept>
#include <string>

// ── Weight context ───────────────────────────────────────────────────────────
// Holds all linear weight tensors. Allocated once at model load, lives until
// wav2vec2_free(). Supports F32, F16, and quantized types (Q8_0, Q4_0, …).

struct GgmlWeights {
    ggml_context * ctx = nullptr;

    GgmlWeights() = default;
    ~GgmlWeights() { if (ctx) { ggml_free(ctx); ctx = nullptr; } }

    GgmlWeights(const GgmlWeights &)            = delete;
    GgmlWeights & operator=(const GgmlWeights &) = delete;

    // Call once before adding any tensors. total_bytes should be the sum of
    // ggml_nbytes() for every tensor you plan to add, plus overhead.
    void init(size_t total_bytes) {
        struct ggml_init_params p = {
            /* .mem_size   = */ total_bytes + 512 * 1024, // 512 KB overhead for headers
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false,
        };
        ctx = ggml_init(p);
        if (!ctx) throw std::runtime_error("ggml_init failed for weight context");
    }

    // Create a 2D tensor of type t with shape [ne0, ne1] and fill it from raw.
    // ne0 = inner dimension (in_features for a weight matrix).
    // ne1 = outer dimension (out_features).
    // raw must be exactly ggml_nbytes(tensor) bytes.
    ggml_tensor * add(ggml_type t, int64_t ne0, int64_t ne1,
                      const void * raw, size_t raw_bytes) {
        auto * tensor = ggml_new_tensor_2d(ctx, t, ne0, ne1);
        if (ggml_nbytes(tensor) != raw_bytes)
            throw std::runtime_error("ggml tensor size mismatch");
        memcpy(tensor->data, raw, raw_bytes);
        return tensor;
    }

    // Convenience: add an F32 tensor from a std::vector<float>.
    ggml_tensor * add_f32(int64_t ne0, int64_t ne1, const std::vector<float> & v) {
        size_t expected = (size_t)ne0 * ne1 * sizeof(float);
        if (v.size() * sizeof(float) != expected)
            throw std::runtime_error("add_f32 size mismatch");
        return add(GGML_TYPE_F32, ne0, ne1, v.data(), expected);
    }
};

// ── Compute scratch ──────────────────────────────────────────────────────────
// A reusable scratch buffer for building per-call compute graphs.
// Allocate once per inference run (sized for the largest single matmul),
// then reuse it for every wv2_linear_ggml call.

struct GgmlScratch {
    std::vector<uint8_t> buf;

    // min_bytes: largest M*K*4 + M*N*4 + overhead you expect per call.
    explicit GgmlScratch(size_t min_bytes = 64 * 1024 * 1024)
        : buf(min_bytes) {}

    // y [M, N] = x [M, K] @ W^T [K, N] + bias [N]
    // W is a ggml_tensor (may be quantized). x and result are F32.
    //
    // ggml convention: ggml_mul_mat(A, B) computes B * A^T in standard math.
    // So ggml_mul_mat(W, x_t) where W=[K,N] and x_t=[K,M] → result=[N,M]=x@W^T. ✓
    std::vector<float> linear(ggml_tensor * W,
                               const float * x, int M, int K, int N,
                               const std::vector<float> & bias,
                               int n_threads) {
        // Size check: make sure scratch is large enough
        size_t needed = (size_t)(M * K + M * N) * sizeof(float)
                      + 8 * ggml_tensor_overhead()
                      + ggml_graph_overhead()
                      + 64 * 1024; // overhead margin
        if (buf.size() < needed) buf.resize(needed * 2);

        struct ggml_init_params p = {
            /* .mem_size   = */ buf.size(),
            /* .mem_buffer = */ buf.data(),
            /* .no_alloc   = */ false,
        };
        ggml_context * ctx = ggml_init(p);

        // Input x: [M, K] → ggml tensor with ne[0]=K (contiguous), ne[1]=M
        ggml_tensor * x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
        memcpy(x_t->data, x, (size_t)M * K * sizeof(float));

        // ggml_mul_mat(W, x_t): W ne[0]=K,ne[1]=N, x_t ne[0]=K,ne[1]=M
        //   → result ne[0]=N, ne[1]=M  stored row-major as [M rows, N cols] ✓
        ggml_tensor * result = ggml_mul_mat(ctx, W, x_t);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, result);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);

        std::vector<float> y((size_t)M * N);
        memcpy(y.data(), result->data, (size_t)M * N * sizeof(float));

        ggml_free(ctx); // does NOT free buf.data() since we provided it

        // Add bias
        if (!bias.empty())
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++)
                    y[(size_t)m * N + n] += bias[n];

        return y;
    }
};
