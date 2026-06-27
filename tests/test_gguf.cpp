// Tests for the GGUF reader: write a minimal GGUF in memory, read it back.
#include "test_utils.h"
#include "../src/gguf.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unistd.h>

// Build a minimal valid GGUF file in a temp file and parse it back
static std::string make_temp_gguf() {
    char path[] = "/tmp/wav2vec2_test_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); exit(1); }
    close(fd);
    FILE * f = fopen(path, "wb");

    auto w1 = [&](uint8_t  v){ fwrite(&v,1,1,f); };
    auto w2 = [&](uint16_t v){ fwrite(&v,2,1,f); };
    auto w4 = [&](uint32_t v){ fwrite(&v,4,1,f); };
    auto w8 = [&](uint64_t v){ fwrite(&v,8,1,f); };
    auto ws = [&](const char * s){
        uint64_t n = strlen(s); w8(n); fwrite(s,1,n,f);
    };

    // Header
    w4(0x46554747u); // magic
    w4(3);           // version
    w8(2);           // n_tensors
    w8(4);           // n_kv

    // KV 1: uint32
    ws("mykey.u32"); w4(4/*UINT32*/); w4(42u);
    // KV 2: string
    ws("mykey.str"); w4(8/*STRING*/); ws("hello");
    // KV 3: array of uint32
    ws("mykey.arr_u32"); w4(9/*ARRAY*/); w4(4/*UINT32*/); w8(3); w4(10u); w4(20u); w4(30u);
    // KV 4: array of string
    ws("mykey.arr_str"); w4(9/*ARRAY*/); w4(8/*STRING*/); w8(2); ws("foo"); ws("bar");

    // Tensor info
    // tensor 0: shape [3], F32
    ws("t0"); w4(1); w8(3); w4(0/*F32*/); w8(0); // offset 0
    // tensor 1: shape [2,2], F16 — 4 elements = 8 bytes; offset = 3*4=12, aligned to 32 → 32
    ws("t1"); w4(2); w8(2); w8(2); w4(1/*F16*/); w8(32);

    // Align to 32 bytes
    long pos = ftell(f);
    int pad = (32 - (pos % 32)) % 32;
    for (int i = 0; i < pad; i++) w1(0);

    // Tensor data
    // t0: [1.0, 2.0, 3.0] as F32
    float t0[] = {1.f, 2.f, 3.f};
    fwrite(t0, 4, 3, f);
    // pad to 32-byte alignment (12 bytes → 20 bytes pad)
    for (int i = 0; i < 20; i++) w1(0);
    // t1: [[1,2],[3,4]] as F16 at offset 32
    uint16_t t1[] = {0x3c00, 0x4000, 0x4200, 0x4400}; // 1.0, 2.0, 3.0, 4.0 in f16
    fwrite(t1, 2, 4, f);

    fclose(f);
    return std::string(path);
}

static void test_gguf_kv() {
    std::string path = make_temp_gguf();
    GgufFile gf = gguf_load(path.c_str());
    unlink(path.c_str());

    // uint32
    ASSERT_EQ(gguf_u32(gf, "mykey.u32"), 42u);
    // string
    ASSERT(gguf_str(gf, "mykey.str") == "hello");
    // array of uint32
    auto & arr_u32 = gguf_arr_u32(gf, "mykey.arr_u32");
    ASSERT_EQ((int)arr_u32.size(), 3);
    ASSERT_EQ(arr_u32[0], 10u);
    ASSERT_EQ(arr_u32[1], 20u);
    ASSERT_EQ(arr_u32[2], 30u);
    // array of string
    auto & arr_str = gguf_arr_str(gf, "mykey.arr_str");
    ASSERT_EQ((int)arr_str.size(), 2);
    ASSERT(arr_str[0] == "foo");
    ASSERT(arr_str[1] == "bar");
}

static void test_gguf_tensors_f32() {
    std::string path = make_temp_gguf();
    GgufFile gf = gguf_load(path.c_str());
    unlink(path.c_str());

    const float * t0 = gguf_tensor(gf, "t0");
    ASSERT_NEAR(t0[0], 1.f, 1e-6f);
    ASSERT_NEAR(t0[1], 2.f, 1e-6f);
    ASSERT_NEAR(t0[2], 3.f, 1e-6f);
}

static void test_gguf_tensors_f16() {
    std::string path = make_temp_gguf();
    GgufFile gf = gguf_load(path.c_str());
    unlink(path.c_str());

    // t1 stored as F16, should be converted to F32 on load
    const float * t1 = gguf_tensor(gf, "t1");
    ASSERT_NEAR(t1[0], 1.f, 1e-3f);
    ASSERT_NEAR(t1[1], 2.f, 1e-3f);
    ASSERT_NEAR(t1[2], 3.f, 1e-3f);
    ASSERT_NEAR(t1[3], 4.f, 1e-3f);
}

static void test_gguf_missing_key_throws() {
    std::string path = make_temp_gguf();
    GgufFile gf = gguf_load(path.c_str());
    unlink(path.c_str());

    bool threw = false;
    try { gguf_u32(gf, "does.not.exist"); }
    catch (const std::runtime_error &) { threw = true; }
    ASSERT(threw);
}

static void test_f16_to_f32_known_values() {
    // Test half-precision conversion for known values
    ASSERT_NEAR(f16_to_f32(0x0000), 0.f,     1e-6f); // +0
    ASSERT_NEAR(f16_to_f32(0x3c00), 1.f,     1e-4f); // 1.0
    ASSERT_NEAR(f16_to_f32(0x4000), 2.f,     1e-4f); // 2.0
    ASSERT_NEAR(f16_to_f32(0xbc00), -1.f,    1e-4f); // -1.0
    ASSERT_NEAR(f16_to_f32(0x7c00), INFINITY, 1e-6f); // +inf
    // NaN check via bit pattern (isnan unreliable with -ffast-math)
    float nan_val = f16_to_f32(0x7e00);
    uint32_t nan_bits; memcpy(&nan_bits, &nan_val, 4);
    ASSERT((nan_bits & 0x7f800000u) == 0x7f800000u && (nan_bits & 0x007fffffu) != 0);
}

int main() {
    fprintf(stderr, "=== gguf tests ===\n");
    test_f16_to_f32_known_values();
    test_gguf_kv();
    test_gguf_tensors_f32();
    test_gguf_tensors_f16();
    test_gguf_missing_key_throws();
    TEST_SUITE_RESULT();
}
