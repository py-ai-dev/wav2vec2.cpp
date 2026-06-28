// Minimal GGUF reader — supports F32, F16, and quantized tensors (Q8_0, Q4_0, …).
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>

// ── half → float ────────────────────────────────────────────────────────────
static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1u;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    if (e == 0)  { float f = m / 1024.0f / 32768.0f; return s ? -f : f; }
    if (e == 31) { return m ? NAN : (s ? -INFINITY : INFINITY); }
    uint32_t bits = (s << 31) | ((e + 112u) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

// ── Types ────────────────────────────────────────────────────────────────────
enum GgufValType : uint32_t {
    GVT_UINT8=0, GVT_INT8, GVT_UINT16, GVT_INT16,
    GVT_UINT32, GVT_INT32, GVT_FLOAT32, GVT_BOOL,
    GVT_STRING, GVT_ARRAY, GVT_UINT64, GVT_INT64, GVT_FLOAT64
};

// GGML tensor type values (matches enum ggml_type in ggml.h).
// We store the raw uint32 so we don't need to include ggml.h here.
enum GgufTensorType : uint32_t {
    GTT_F32  = 0,
    GTT_F16  = 1,
    GTT_Q4_0 = 2,
    GTT_Q4_1 = 3,
    GTT_Q5_0 = 6,
    GTT_Q5_1 = 7,
    GTT_Q8_0 = 8,
    GTT_Q8_1 = 9,
};

// Returns the number of raw bytes for n elements of a given quantized type.
// Quantization block sizes match ggml's internal layout (GGUF uses the same format).
static inline size_t gguf_tensor_nbytes(GgufTensorType t, uint64_t n_elems) {
    switch (t) {
        case GTT_F32:  return n_elems * 4;
        case GTT_F16:  return n_elems * 2;
        // Q4_0: 32-element blocks, each block = 2 bytes (f16 scale) + 16 bytes (nibbles)
        case GTT_Q4_0: return (n_elems / 32) * 18;
        // Q4_1: 32-element blocks = 2+2 bytes (f16 scale+min) + 16 bytes
        case GTT_Q4_1: return (n_elems / 32) * 20;
        // Q5_0: 32-element blocks = 2 bytes (f16 scale) + 4 bytes (high bits) + 16 bytes (nibbles)
        case GTT_Q5_0: return (n_elems / 32) * 22;
        // Q5_1: 32-element blocks = 2+2 bytes + 4 bytes + 16 bytes
        case GTT_Q5_1: return (n_elems / 32) * 24;
        // Q8_0: 32-element blocks = 2 bytes (f16 scale) + 32 bytes (int8)
        case GTT_Q8_0: return (n_elems / 32) * 34;
        // Q8_1: 32-element blocks = 2+2 bytes + 32 bytes
        case GTT_Q8_1: return (n_elems / 32) * 36;
        default: return n_elems * 4; // unknown: assume F32
    }
}

struct GgufTensor {
    std::vector<uint64_t> shape;   // dims, innermost first (as in GGUF)
    uint64_t              n_elems; // total element count
    GgufTensorType        gtype;   // original GGML type
    std::vector<float>    data;    // F32 values (populated for F32 and F16 types only)
    std::vector<uint8_t>  raw;     // raw bytes (populated for ALL types, incl. quantized)
};

struct GgufKV {
    GgufValType              type;
    uint64_t                 u64 = 0;
    float                    f32 = 0.f;
    std::string              str;
    std::vector<std::string> arr_str;
    std::vector<uint32_t>    arr_u32;
};

struct GgufFile {
    std::unordered_map<std::string, GgufKV>     kv;
    std::unordered_map<std::string, GgufTensor> tensors;
};

// ── Reader ───────────────────────────────────────────────────────────────────
static std::string gguf_read_str(std::ifstream & f) {
    uint64_t len; f.read((char*)&len, 8);
    std::string s(len, '\0'); f.read(s.data(), (std::streamsize)len);
    return s;
}

static void gguf_read_scalar(std::ifstream & f, GgufValType t, GgufKV & kv) {
    switch (t) {
        case GVT_UINT8:   { uint8_t  v; f.read((char*)&v,1); kv.u64=v; break; }
        case GVT_INT8:    { int8_t   v; f.read((char*)&v,1); kv.u64=(uint64_t)v; break; }
        case GVT_UINT16:  { uint16_t v; f.read((char*)&v,2); kv.u64=v; break; }
        case GVT_INT16:   { int16_t  v; f.read((char*)&v,2); kv.u64=(uint64_t)v; break; }
        case GVT_UINT32:  { uint32_t v; f.read((char*)&v,4); kv.u64=v; break; }
        case GVT_INT32:   { int32_t  v; f.read((char*)&v,4); kv.u64=(uint64_t)v; break; }
        case GVT_UINT64:  { f.read((char*)&kv.u64, 8); break; }
        case GVT_INT64:   { int64_t  v; f.read((char*)&v,8); kv.u64=(uint64_t)v; break; }
        case GVT_FLOAT32: { f.read((char*)&kv.f32, 4); break; }
        case GVT_FLOAT64: { double v; f.read((char*)&v,8); kv.f32=(float)v; break; }
        case GVT_BOOL:    { uint8_t v; f.read((char*)&v,1); kv.u64=v; break; }
        case GVT_STRING:  { kv.str = gguf_read_str(f); break; }
        default: throw std::runtime_error("gguf: unsupported scalar type " + std::to_string(t));
    }
}

static GgufFile gguf_load(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);

    uint32_t magic, version; uint64_t n_tensors, n_kv;
    f.read((char*)&magic,   4); if (magic != 0x46554747u) throw std::runtime_error("Not GGUF");
    f.read((char*)&version, 4);
    f.read((char*)&n_tensors, 8);
    f.read((char*)&n_kv,      8);

    GgufFile gf;

    // Key-value metadata
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = gguf_read_str(f);
        uint32_t vt; f.read((char*)&vt, 4);
        GgufKV kv; kv.type = (GgufValType)vt;
        if (kv.type == GVT_ARRAY) {
            uint32_t et; uint64_t an;
            f.read((char*)&et, 4); f.read((char*)&an, 8);
            GgufValType elem_t = (GgufValType)et;
            for (uint64_t j = 0; j < an; j++) {
                if (elem_t == GVT_STRING)      { kv.arr_str.push_back(gguf_read_str(f)); }
                else if (elem_t == GVT_UINT32) { uint32_t v; f.read((char*)&v,4); kv.arr_u32.push_back(v); }
                else { GgufKV tmp; tmp.type=elem_t; gguf_read_scalar(f, elem_t, tmp); /* discard */ }
            }
        } else {
            gguf_read_scalar(f, kv.type, kv);
        }
        gf.kv[key] = std::move(kv);
    }

    // Tensor info
    struct TInfo {
        std::string         name;
        std::vector<uint64_t> shape;
        GgufTensorType      type;
        uint64_t            offset;
    };
    std::vector<TInfo> tinfos(n_tensors);
    for (auto & ti : tinfos) {
        ti.name = gguf_read_str(f);
        uint32_t nd; f.read((char*)&nd, 4);
        ti.shape.resize(nd);
        for (auto & d : ti.shape) f.read((char*)&d, 8);
        uint32_t tt; f.read((char*)&tt, 4); ti.type = (GgufTensorType)tt;
        f.read((char*)&ti.offset, 8);
    }

    // Align to 32 bytes for data section
    uint64_t pos = f.tellg();
    uint64_t data_start = (pos + 31u) & ~31ull;
    f.seekg((std::streamoff)data_start);

    // Load tensors
    for (auto & ti : tinfos) {
        f.seekg((std::streamoff)(data_start + ti.offset));
        uint64_t n = 1; for (auto d : ti.shape) n *= d;
        size_t nbytes = gguf_tensor_nbytes(ti.type, n);

        GgufTensor gt;
        gt.shape   = ti.shape;
        gt.n_elems = n;
        gt.gtype   = ti.type;

        // Always load raw bytes
        gt.raw.resize(nbytes);
        f.read((char*)gt.raw.data(), (std::streamsize)nbytes);

        // For F32/F16, also populate the float32 data array for backward compatibility
        if (ti.type == GTT_F32) {
            gt.data.resize(n);
            memcpy(gt.data.data(), gt.raw.data(), n * 4);
        } else if (ti.type == GTT_F16) {
            gt.data.resize(n);
            const uint16_t * src = (const uint16_t *)gt.raw.data();
            for (uint64_t j = 0; j < n; j++) gt.data[j] = f16_to_f32(src[j]);
        }

        gf.tensors[ti.name] = std::move(gt);
    }
    return gf;
}

// Accessors
static const float * gguf_tensor(const GgufFile & g, const std::string & n) {
    auto it = g.tensors.find(n);
    if (it == g.tensors.end()) throw std::runtime_error("Missing tensor: " + n);
    if (it->second.data.empty())
        throw std::runtime_error("Tensor " + n + " is quantized — use raw bytes + ggml");
    return it->second.data.data();
}
static uint32_t gguf_u32(const GgufFile & g, const std::string & k) {
    auto it = g.kv.find(k);
    if (it == g.kv.end()) throw std::runtime_error("Missing key: " + k);
    return (uint32_t)it->second.u64;
}
static std::string gguf_str(const GgufFile & g, const std::string & k) {
    auto it = g.kv.find(k);
    if (it == g.kv.end()) throw std::runtime_error("Missing key: " + k);
    return it->second.str;
}
static const std::vector<std::string> & gguf_arr_str(const GgufFile & g, const std::string & k) {
    auto it = g.kv.find(k);
    if (it == g.kv.end()) throw std::runtime_error("Missing key: " + k);
    return it->second.arr_str;
}
static const std::vector<uint32_t> & gguf_arr_u32(const GgufFile & g, const std::string & k) {
    auto it = g.kv.find(k);
    if (it == g.kv.end()) throw std::runtime_error("Missing key: " + k);
    return it->second.arr_u32;
}
