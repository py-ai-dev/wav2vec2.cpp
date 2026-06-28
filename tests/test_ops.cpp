// Unit tests for math primitives in src/ops.h
#include "test_utils.h"
#include "../src/ops.h"
#include <vector>
#include <cmath>

// ── gelu ────────────────────────────────────────────────────────────────────
static void test_gelu() {
    // gelu(0) = 0
    ASSERT_NEAR(wv2_gelu(0.f), 0.f, 1e-6f);
    // gelu(x) ≈ x for large positive x
    ASSERT_NEAR(wv2_gelu(10.f), 10.f, 1e-3f);
    // gelu is roughly half for negative (soft gate)
    ASSERT(wv2_gelu(-1.f) > -1.f && wv2_gelu(-1.f) < 0.f);
    // Known value: gelu(1.0) ≈ 0.8413
    ASSERT_NEAR(wv2_gelu(1.f), 0.8413f, 1e-3f);
}

// ── softmax ──────────────────────────────────────────────────────────────────
static void test_softmax() {
    float x[] = {1.f, 2.f, 3.f};
    wv2_softmax(x, 3);
    // Outputs sum to 1
    ASSERT_NEAR(x[0] + x[1] + x[2], 1.f, 1e-6f);
    // All positive
    ASSERT(x[0] > 0.f && x[1] > 0.f && x[2] > 0.f);
    // Largest input → largest output
    ASSERT(x[2] > x[1] && x[1] > x[0]);

    // Uniform inputs → uniform outputs
    float y[] = {2.f, 2.f, 2.f, 2.f};
    wv2_softmax(y, 4);
    ASSERT_NEAR(y[0], 0.25f, 1e-6f);
    ASSERT_NEAR(y[3], 0.25f, 1e-6f);

    // Numerical stability: large values shouldn't produce nan/inf
    float z[] = {1000.f, 1001.f, 999.f};
    wv2_softmax(z, 3);
    ASSERT_NEAR(z[0] + z[1] + z[2], 1.f, 1e-5f);
    ASSERT(z[1] > z[0] && z[0] > z[2]);
}

// ── layer_norm ───────────────────────────────────────────────────────────────
static void test_layer_norm() {
    // Single row [1, 4]: after LN, mean≈0 and std≈1 (before affine)
    float x[] = {1.f, 2.f, 3.f, 4.f};
    float w[] = {1.f, 1.f, 1.f, 1.f}; // identity affine
    float b[] = {0.f, 0.f, 0.f, 0.f};
    wv2_layer_norm(x, w, b, 1, 4);
    float mean = (x[0]+x[1]+x[2]+x[3]) / 4.f;
    ASSERT_NEAR(mean, 0.f, 1e-5f);
    float var = 0.f;
    for (int i = 0; i < 4; i++) var += x[i]*x[i];
    ASSERT_NEAR(sqrtf(var/4.f), 1.f, 1e-3f);

    // Affine scale+shift
    float x2[] = {0.f, 0.f, 0.f, 0.f}; // constant input — LN output is 0 * w + b = b
    float w2[] = {2.f, 2.f, 2.f, 2.f};
    float b2[] = {5.f, 5.f, 5.f, 5.f};
    wv2_layer_norm(x2, w2, b2, 1, 4);
    // All same → mean=0, var=0, but eps prevents div-by-zero; output should equal b
    for (int i = 0; i < 4; i++) ASSERT_NEAR(x2[i], 5.f, 1e-3f);

    // Two rows: each normalised independently
    float x3[] = {10.f, 20.f, 10.f, 20.f,   // row 0
                   1.f,  1.f,  1.f,  1.f};   // row 1 (constant)
    float w3[] = {1.f, 1.f, 1.f, 1.f};
    float b3[] = {0.f, 0.f, 0.f, 0.f};
    wv2_layer_norm(x3, w3, b3, 2, 4);
    float mean0 = (x3[0]+x3[1]+x3[2]+x3[3]) / 4.f;
    ASSERT_NEAR(mean0, 0.f, 1e-5f);
}

// ── group_norm ───────────────────────────────────────────────────────────────
static void test_group_norm() {
    // 3 time steps, 2 channels
    // channel 0: [1, 2, 3] → mean=2, std=√(2/3)
    // channel 1: [4, 4, 4] → mean=4, std≈0 → output≈0*w+b=0 (due to eps)
    float x[] = {1.f,4.f,  2.f,4.f,  3.f,4.f}; // [T=3, C=2] interleaved
    float w[] = {1.f, 1.f};
    float b[] = {0.f, 0.f};
    wv2_group_norm(x, w, b, 3, 2);
    // Channel 0 should be normalised (mean≈0 over T)
    float mean_c0 = (x[0] + x[2] + x[4]) / 3.f;
    ASSERT_NEAR(mean_c0, 0.f, 1e-5f);
    // Channel 1 was constant → output ≈ 0
    ASSERT_NEAR(fabsf(x[1]), 0.f, 1e-2f);
    ASSERT_NEAR(fabsf(x[3]), 0.f, 1e-2f);
}

// ── linear ───────────────────────────────────────────────────────────────────
static void test_linear() {
    // x [2,3] @ W^T [2,3] + b [2]
    // W = [[1,0,0],[0,1,0]]  → selects first two channels
    float x[] = {1.f,2.f,3.f,  4.f,5.f,6.f};
    std::vector<float> W = {1.f,0.f,0.f,  0.f,1.f,0.f}; // [2,3]
    std::vector<float> b = {10.f, 20.f};
    auto y = wv2_linear(x, W, b, 2, 3, 2);
    ASSERT_NEAR(y[0], 1.f + 10.f, 1e-6f);  // row 0, out 0
    ASSERT_NEAR(y[1], 2.f + 20.f, 1e-6f);  // row 0, out 1
    ASSERT_NEAR(y[2], 4.f + 10.f, 1e-6f);  // row 1, out 0
    ASSERT_NEAR(y[3], 5.f + 20.f, 1e-6f);  // row 1, out 1
}

// ── conv1d ───────────────────────────────────────────────────────────────────
static void test_conv1d() {
    // Identity conv: kernel=[1], stride=1, Cin=Cout=1
    // x [5, 1], output should equal input
    float x[] = {1.f, 2.f, 3.f, 4.f, 5.f};
    float w[] = {1.f}; // [1,1,1]
    float b[] = {0.f};
    auto y = wv2_conv1d(x, w, b, 5, 1, 1, 1, 1, 0);
    ASSERT_EQ((int)y.size(), 5);
    for (int i = 0; i < 5; i++) ASSERT_NEAR(y[i], x[i], 1e-6f);

    // Sum-pool kernel=[1,1], stride=1: each output = x[t]+x[t+1]
    float x2[] = {1.f, 2.f, 3.f, 4.f};
    float w2[] = {1.f, 1.f}; // [1,1,2]
    float b2[] = {0.f};
    auto y2 = wv2_conv1d(x2, w2, b2, 4, 1, 1, 2, 1, 0);
    ASSERT_EQ((int)y2.size(), 3);
    ASSERT_NEAR(y2[0], 3.f, 1e-6f); // 1+2
    ASSERT_NEAR(y2[1], 5.f, 1e-6f); // 2+3
    ASSERT_NEAR(y2[2], 7.f, 1e-6f); // 3+4

    // Stride 2
    auto y3 = wv2_conv1d(x2, w2, b2, 4, 1, 1, 2, 2, 0);
    ASSERT_EQ((int)y3.size(), 2);
    ASSERT_NEAR(y3[0], 3.f, 1e-6f);
    ASSERT_NEAR(y3[1], 7.f, 1e-6f);

    // Padding: pad=1 on each side of x=[1,2,3], kernel=3 → output length 3
    float x3[] = {1.f, 2.f, 3.f};
    float w3[] = {1.f, 1.f, 1.f}; // sum of window
    float b3[] = {0.f};
    auto y4 = wv2_conv1d(x3, w3, b3, 3, 1, 1, 3, 1, 1);
    ASSERT_EQ((int)y4.size(), 3);
    ASSERT_NEAR(y4[0], 0.f+1.f+2.f, 1e-6f); // [pad,1,2]
    ASSERT_NEAR(y4[1], 1.f+2.f+3.f, 1e-6f); // [1,2,3]
    ASSERT_NEAR(y4[2], 2.f+3.f+0.f, 1e-6f); // [2,3,pad]
}

// ── conv1d output length formula ─────────────────────────────────────────────
static void test_feature_extractor_output_length() {
    // wav2vec2-base: 7 conv layers with known strides/kernels
    // 1 second at 16000 Hz → ~49 frames
    int L = 16000;
    int kernels[] = {10, 3, 3, 3, 3, 2, 2};
    int strides[] = {5,  2, 2, 2, 2, 2, 2};
    for (int i = 0; i < 7; i++)
        L = (L - kernels[i]) / strides[i] + 1;
    // Should be ~49 frames
    ASSERT(L >= 45 && L <= 55);
}

// ── ctc_greedy ───────────────────────────────────────────────────────────────
static void test_ctc_greedy() {
    // vocab_size=4, blank=0
    // logits: T=6, each row is scores over 4 tokens
    // pattern: blank, 1, 1, blank, 2, blank → decode → [1, 2]
    float logits[6*4] = {
        10,0,0,0,  // t=0: blank
        0,10,0,0,  // t=1: token 1
        0,10,0,0,  // t=2: token 1 (dup → collapsed)
        10,0,0,0,  // t=3: blank
        0,0,10,0,  // t=4: token 2
        10,0,0,0,  // t=5: blank
    };
    auto ids = wv2_ctc_greedy(logits, 6, 4);
    // raw greedy: [0,1,1,0,2,0] → collapse dups → [0,1,0,2,0]
    ASSERT_EQ((int)ids.size(), 5);
    ASSERT_EQ(ids[0], 0);
    ASSERT_EQ(ids[1], 1);
    ASSERT_EQ(ids[2], 0);
    ASSERT_EQ(ids[3], 2);
    ASSERT_EQ(ids[4], 0);

    // All same token → single id
    float logits2[3*2] = {0,10,  0,10,  0,10};
    auto ids2 = wv2_ctc_greedy(logits2, 3, 2);
    ASSERT_EQ((int)ids2.size(), 1);
    ASSERT_EQ(ids2[0], 1);
}

// ── ctc_beam_search ──────────────────────────────────────────────────────────
static void test_ctc_beam_search() {
    // Same scenario as greedy: blank, 1, 1, blank, 2, blank → [1, 2]
    float logits[6*4] = {
        10,0,0,0,  // t=0: blank
        0,10,0,0,  // t=1: token 1
        0,10,0,0,  // t=2: token 1 (dup)
        10,0,0,0,  // t=3: blank
        0,0,10,0,  // t=4: token 2
        10,0,0,0,  // t=5: blank
    };
    auto ids = wv2_ctc_beam_search(logits, 6, 4, 0, 5);
    ASSERT_EQ((int)ids.size(), 2);
    ASSERT_EQ(ids[0], 1);
    ASSERT_EQ(ids[1], 2);

    // Two same tokens separated by blank: [1, blank, 1] → [1, 1]
    float logits2[3*3] = {
        0,10,0,   // token 1
        10,0,0,   // blank
        0,10,0,   // token 1
    };
    auto ids2 = wv2_ctc_beam_search(logits2, 3, 3, 0, 5);
    ASSERT_EQ((int)ids2.size(), 2);
    ASSERT_EQ(ids2[0], 1);
    ASSERT_EQ(ids2[1], 1);

    // All blanks → empty
    float logits3[4*2] = {
        10,0,  10,0,  10,0,  10,0,
    };
    auto ids3 = wv2_ctc_beam_search(logits3, 4, 2, 0, 5);
    ASSERT_EQ((int)ids3.size(), 0);

    // Single repeated token (no blank) → single token
    float logits4[3*2] = {
        0,10,  0,10,  0,10,
    };
    auto ids4 = wv2_ctc_beam_search(logits4, 3, 2, 0, 5);
    ASSERT_EQ((int)ids4.size(), 1);
    ASSERT_EQ(ids4[0], 1);
}

int main() {
    fprintf(stderr, "=== ops tests ===\n");
    test_gelu();
    test_softmax();
    test_layer_norm();
    test_group_norm();
    test_linear();
    test_conv1d();
    test_feature_extractor_output_length();
    test_ctc_greedy();
    test_ctc_beam_search();
    TEST_SUITE_RESULT();
}
