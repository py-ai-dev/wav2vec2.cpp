#!/usr/bin/env python3
"""
Convert any HuggingFace Wav2Vec2ForCTC model to GGUF format.

Usage:
    python convert_to_gguf.py facebook/wav2vec2-base-960h output.gguf
    python convert_to_gguf.py facebook/wav2vec2-base-960h output.gguf --dtype f16
    python convert_to_gguf.py facebook/wav2vec2-base-960h output.gguf --dtype q8_0
    python convert_to_gguf.py facebook/wav2vec2-base-960h output.gguf --dtype q4_0

Tensor routing by dtype:
  f32  — all tensors in F32 (largest, most precise)
  f16  — all tensors in F16 (~half size, near-lossless)
  q8_0 — linear weights quantized to 8-bit; norms/biases/convs stay F32
  q4_0 — linear weights quantized to 4-bit; norms/biases/convs stay F32
"""

import sys
import struct
import argparse
import numpy as np
from pathlib import Path

# ── GGUF type constants (match ggml_type enum) ───────────────────────────────
GGUF_F32  = 0
GGUF_F16  = 1
GGUF_Q4_0 = 2
GGUF_Q8_0 = 8

# ── GGUF wire helpers ─────────────────────────────────────────────────────────

def write_str(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_kv_scalar(f, key: str, val, val_type: int):
    write_str(f, key)
    f.write(struct.pack("<I", val_type))
    if val_type == 4:   f.write(struct.pack("<I", val))
    elif val_type == 6: f.write(struct.pack("<f", val))
    elif val_type == 8: write_str(f, val)
    elif val_type == 7: f.write(struct.pack("<?", val))


def write_kv_arr_str(f, key: str, arr):
    write_str(f, key)
    f.write(struct.pack("<II", 9, 8))          # ARRAY of STRING
    f.write(struct.pack("<Q", len(arr)))
    for s in arr: write_str(f, s)


def write_kv_arr_u32(f, key: str, arr):
    write_str(f, key)
    f.write(struct.pack("<II", 9, 4))          # ARRAY of UINT32
    f.write(struct.pack("<Q", len(arr)))
    for v in arr: f.write(struct.pack("<I", v))


# ── Quantization ─────────────────────────────────────────────────────────────

def quantize_q8_0(arr: np.ndarray) -> bytes:
    """Q8_0: blocks of 32, scale stored as f16, values as int8."""
    flat = arr.flatten().astype(np.float32)
    n = len(flat)
    assert n % 32 == 0, f"Q8_0 requires element count divisible by 32, got {n}"
    blocks = flat.reshape(-1, 32)
    out = bytearray()
    for blk in blocks:
        d = np.max(np.abs(blk))
        scale = d / 127.0 if d > 0 else 1.0
        iscale = 1.0 / scale
        qs = np.clip(np.round(blk * iscale), -128, 127).astype(np.int8)
        out += struct.pack("<e", scale)   # f16 scale
        out += qs.tobytes()
    return bytes(out)


def quantize_q4_0(arr: np.ndarray) -> bytes:
    """Q4_0: blocks of 32, scale stored as f16, values as 4-bit unsigned packed 2/byte."""
    flat = arr.flatten().astype(np.float32)
    n = len(flat)
    assert n % 32 == 0, f"Q4_0 requires element count divisible by 32, got {n}"
    blocks = flat.reshape(-1, 32)
    out = bytearray()
    for blk in blocks:
        d = np.max(np.abs(blk))
        scale = d / 7.0 if d > 0 else 1.0
        iscale = 1.0 / scale
        # Quantize to [0..15] unsigned by adding 8
        qs = np.clip(np.round(blk * iscale) + 8, 0, 15).astype(np.uint8)
        out += struct.pack("<e", scale)   # f16 scale
        # ggml Q4_0 layout: byte j = low nibble: element j, high nibble: element j+16
        packed = qs[:16] | (qs[16:] << 4)
        out += packed.tobytes()
    return bytes(out)


# ── Main conversion ───────────────────────────────────────────────────────────

# Names of linear weight matrices (candidates for quantization).
# Everything else (norms, biases, conv weights) stays in F32.
def is_linear_weight(name: str) -> bool:
    return name.endswith(".weight") and any(k in name for k in [
        "projection.weight",
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight",
        "intermediate_dense.weight", "output_dense.weight",
        "lm_head.weight",
    ])


def encode_tensor(name: str, arr: np.ndarray, dtype: str):
    """Return (gguf_type, raw_bytes) for a tensor given the target dtype."""
    if dtype in ("q8_0", "q4_0") and is_linear_weight(name):
        n = arr.size
        pad = (32 - n % 32) % 32
        if pad:
            arr = np.append(arr.flatten(), np.zeros(pad, dtype=np.float32)).reshape(-1)
        if dtype == "q8_0":
            return GGUF_Q8_0, quantize_q8_0(arr)
        else:
            return GGUF_Q4_0, quantize_q4_0(arr)
    elif dtype == "f16":
        return GGUF_F16, arr.astype(np.float16).tobytes()
    else:
        return GGUF_F32, arr.astype(np.float32).tobytes()


def convert(model_id: str, out_path: str, dtype: str = "f16"):
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    except ImportError:
        print("pip install transformers torch"); sys.exit(1)

    import torch
    print(f"Loading {model_id} ...")
    model     = Wav2Vec2ForCTC.from_pretrained(model_id)
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model.eval()

    cfg = model.config
    tok = processor.tokenizer

    vocab_dict = tok.get_vocab()
    vocab = [""] * len(vocab_dict)
    for token, idx in vocab_dict.items():
        if idx < len(vocab):
            vocab[idx] = token

    pad_id          = tok.pad_token_id if tok.pad_token_id is not None else 0
    feat_norm       = getattr(cfg, "feat_extract_norm", "group")
    conv_kernel     = list(cfg.conv_kernel)
    conv_stride     = list(cfg.conv_stride)
    pos_conv_kernel = cfg.num_conv_pos_embeddings
    pos_conv_groups = cfg.num_conv_pos_embedding_groups
    n_conv          = len(conv_kernel)
    n_layers        = cfg.num_hidden_layers

    print(f"  layers={n_layers}  hidden={cfg.hidden_size}  "
          f"heads={cfg.num_attention_heads}  vocab={len(vocab)}  dtype={dtype}")

    tensors = {}  # name → np.ndarray (F32)

    def add(name, t):
        import torch as _torch
        if isinstance(t, _torch.Tensor):
            t = t.detach().float().numpy()
        tensors[name] = t.astype(np.float32)

    sd = model.state_dict()

    # CNN feature extractor
    for i in range(n_conv):
        p  = f"wav2vec2.feature_extractor.conv_layers.{i}"
        op = f"feature_extractor.conv_layers.{i}"
        add(f"{op}.conv.weight", sd[f"{p}.conv.weight"])
        if f"{p}.conv.bias" in sd:
            add(f"{op}.conv.bias", sd[f"{p}.conv.bias"])
        if f"{p}.layer_norm.weight" in sd:
            add(f"{op}.layer_norm.weight", sd[f"{p}.layer_norm.weight"])
            add(f"{op}.layer_norm.bias",   sd[f"{p}.layer_norm.bias"])

    # Feature projection
    add("feature_projection.layer_norm.weight", sd["wav2vec2.feature_projection.layer_norm.weight"])
    add("feature_projection.layer_norm.bias",   sd["wav2vec2.feature_projection.layer_norm.bias"])
    add("feature_projection.projection.weight", sd["wav2vec2.feature_projection.projection.weight"])
    add("feature_projection.projection.bias",   sd["wav2vec2.feature_projection.projection.bias"])

    # Positional conv — access .weight directly so parametrize/weight_norm is
    # applied transparently regardless of transformers/torch version.
    add("encoder.pos_conv_embed.conv.weight",
        model.wav2vec2.encoder.pos_conv_embed.conv.weight.detach())
    add("encoder.pos_conv_embed.conv.bias",
        model.wav2vec2.encoder.pos_conv_embed.conv.bias.detach())

    # Encoder layer norm
    add("encoder.layer_norm.weight", sd["wav2vec2.encoder.layer_norm.weight"])
    add("encoder.layer_norm.bias",   sd["wav2vec2.encoder.layer_norm.bias"])

    # Transformer layers
    for i in range(n_layers):
        hp = f"wav2vec2.encoder.layers.{i}"
        op = f"encoder.layers.{i}"
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            add(f"{op}.attention.{proj}.weight", sd[f"{hp}.attention.{proj}.weight"])
            add(f"{op}.attention.{proj}.bias",   sd[f"{hp}.attention.{proj}.bias"])
        add(f"{op}.feed_forward.intermediate_dense.weight", sd[f"{hp}.feed_forward.intermediate_dense.weight"])
        add(f"{op}.feed_forward.intermediate_dense.bias",   sd[f"{hp}.feed_forward.intermediate_dense.bias"])
        add(f"{op}.feed_forward.output_dense.weight",       sd[f"{hp}.feed_forward.output_dense.weight"])
        add(f"{op}.feed_forward.output_dense.bias",         sd[f"{hp}.feed_forward.output_dense.bias"])
        add(f"{op}.layer_norm.weight",       sd[f"{hp}.layer_norm.weight"])
        add(f"{op}.layer_norm.bias",         sd[f"{hp}.layer_norm.bias"])
        add(f"{op}.final_layer_norm.weight", sd[f"{hp}.final_layer_norm.weight"])
        add(f"{op}.final_layer_norm.bias",   sd[f"{hp}.final_layer_norm.bias"])

    # CTC head
    add("lm_head.weight", sd["lm_head.weight"])
    if "lm_head.bias" in sd:
        add("lm_head.bias", sd["lm_head.bias"])
    else:
        add("lm_head.bias", np.zeros(len(vocab), dtype=np.float32))

    # ── Encode tensors ───────────────────────────────────────────────────────
    print(f"Quantizing / encoding tensors ...")
    encoded = {}   # name → (gguf_type, bytes)
    n_quantized = 0
    for name, arr in tensors.items():
        gtype, data = encode_tensor(name, arr, dtype)
        encoded[name] = (gtype, data)
        if gtype in (GGUF_Q8_0, GGUF_Q4_0):
            n_quantized += 1

    if n_quantized:
        print(f"  quantized {n_quantized}/{len(tensors)} tensors")

    # ── Write GGUF ───────────────────────────────────────────────────────────
    print(f"Writing {out_path} ...")
    ALIGN = 32
    n_tensors = len(tensors)

    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", 0x46554747))  # magic
        f.write(struct.pack("<I", 3))            # version
        f.write(struct.pack("<Q", n_tensors))
        kv_count_pos = f.tell()
        f.write(struct.pack("<Q", 0))            # placeholder KV count

        kv_written = 0
        def kv_u32(key, val):
            nonlocal kv_written
            write_kv_scalar(f, key, val, 4); kv_written += 1
        def kv_str(key, val):
            nonlocal kv_written
            write_kv_scalar(f, key, val, 8); kv_written += 1

        kv_u32("wav2vec2.n_conv_layers",     n_conv)
        kv_u32("wav2vec2.conv_dim",          cfg.conv_dim[0] if hasattr(cfg.conv_dim, "__len__") else cfg.conv_dim)
        kv_u32("wav2vec2.n_encoder_layers",  n_layers)
        kv_u32("wav2vec2.n_heads",           cfg.num_attention_heads)
        kv_u32("wav2vec2.hidden_size",       cfg.hidden_size)
        kv_u32("wav2vec2.intermediate_size", cfg.intermediate_size)
        kv_u32("wav2vec2.vocab_size",        len(vocab))
        kv_u32("wav2vec2.pad_token_id",      pad_id)
        kv_u32("wav2vec2.pos_conv_kernel",   pos_conv_kernel)
        kv_u32("wav2vec2.pos_conv_groups",   pos_conv_groups)
        kv_str("wav2vec2.feat_extract_norm",   feat_norm)
        kv_str("wav2vec2.word_delimiter_token",
               tok.word_delimiter_token if tok.word_delimiter_token else "|")
        write_kv_scalar(f, "wav2vec2.stable_layer_norm",
                        bool(getattr(cfg, "do_stable_layer_norm", False)), 7)
        kv_written += 1
        write_kv_arr_u32(f, "wav2vec2.conv_kernel", conv_kernel); kv_written += 1
        write_kv_arr_u32(f, "wav2vec2.conv_stride", conv_stride); kv_written += 1
        write_kv_arr_str(f, "tokenizer.ggml.tokens", vocab);      kv_written += 1

        # Patch KV count
        end_pos = f.tell()
        f.seek(kv_count_pos)
        f.write(struct.pack("<Q", kv_written))
        f.seek(end_pos)

        # Tensor info
        cur_offset = 0
        for name, arr in tensors.items():
            gtype, data = encoded[name]
            shape = list(reversed(arr.shape))   # GGUF: innermost first
            write_str(f, name)
            f.write(struct.pack("<I", len(shape)))
            for d in shape: f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", gtype))
            f.write(struct.pack("<Q", cur_offset))
            pad = (ALIGN - (len(data) % ALIGN)) % ALIGN
            cur_offset += len(data) + pad

        # Align before data
        pos = f.tell()
        pad = (ALIGN - (pos % ALIGN)) % ALIGN
        f.write(b"\x00" * pad)

        # Tensor data
        for name in tensors:
            _, data = encoded[name]
            f.write(data)
            pad = (ALIGN - (len(data) % ALIGN)) % ALIGN
            if pad: f.write(b"\x00" * pad)

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Done: {out_path} ({size_mb:.1f} MB, {n_tensors} tensors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Wav2Vec2ForCTC → GGUF")
    parser.add_argument("model",  help="HuggingFace model ID or local path")
    parser.add_argument("output", help="Output .gguf file")
    parser.add_argument("--dtype", choices=["f32", "f16", "q8_0", "q4_0"], default="f16",
                        help="Weight dtype: f32/f16 for all tensors; q8_0/q4_0 quantizes linear layers only")
    args = parser.parse_args()
    convert(args.model, args.output, args.dtype)
