#!/usr/bin/env python3
"""
Convert any HuggingFace Wav2Vec2ForCTC model to GGUF format.

Usage:
    python convert_to_gguf.py facebook/wav2vec2-base-960h output.gguf
    python convert_to_gguf.py liodon-ai/whisper-large-v3-te output.gguf --dtype f16
"""

import sys
import struct
import json
import argparse
import numpy as np
from pathlib import Path


def write_str(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_kv_scalar(f, key: str, val, val_type: int):
    write_str(f, key)
    f.write(struct.pack("<I", val_type))
    if val_type == 4:   f.write(struct.pack("<I", val))       # UINT32
    elif val_type == 6: f.write(struct.pack("<f", val))        # FLOAT32
    elif val_type == 8: write_str(f, val)                      # STRING
    elif val_type == 7: f.write(struct.pack("<?", val))        # BOOL


def write_kv_arr_str(f, key: str, arr):
    write_str(f, key)
    f.write(struct.pack("<I", 9))          # ARRAY
    f.write(struct.pack("<I", 8))          # element type = STRING
    f.write(struct.pack("<Q", len(arr)))
    for s in arr:
        write_str(f, s)


def write_kv_arr_u32(f, key: str, arr):
    write_str(f, key)
    f.write(struct.pack("<I", 9))          # ARRAY
    f.write(struct.pack("<I", 4))          # element type = UINT32
    f.write(struct.pack("<Q", len(arr)))
    for v in arr:
        f.write(struct.pack("<I", v))


def to_f16(tensor: np.ndarray) -> bytes:
    return tensor.astype(np.float16).tobytes()


def to_f32(tensor: np.ndarray) -> bytes:
    return tensor.astype(np.float32).tobytes()


def compute_weight_norm(weight_v, weight_g):
    """Materialise weight from weight_norm decomposition: w = g * v / ||v||"""
    norm = np.linalg.norm(weight_v.reshape(weight_v.shape[0], -1), axis=1)
    norm = norm[:, None, None] if weight_v.ndim == 3 else norm[:, None]
    weight = weight_g * weight_v / (norm + 1e-12)
    return weight


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

    # Build vocabulary list indexed by id
    vocab_dict = tok.get_vocab()
    vocab = [""] * len(vocab_dict)
    for token, idx in vocab_dict.items():
        if idx < len(vocab):
            vocab[idx] = token

    pad_id  = tok.pad_token_id if tok.pad_token_id is not None else 0
    feat_norm = getattr(cfg, "feat_extract_norm", "group")
    conv_kernel = list(cfg.conv_kernel)
    conv_stride = list(cfg.conv_stride)

    pos_conv_kernel = cfg.num_conv_pos_embeddings         # 128
    pos_conv_groups = cfg.num_conv_pos_embedding_groups   # 16

    print(f"  layers={cfg.num_hidden_layers}  hidden={cfg.hidden_size}  "
          f"heads={cfg.num_attention_heads}  vocab={len(vocab)}")
    print(f"  feat_norm={feat_norm}  pos_conv_kernel={pos_conv_kernel}")

    # Collect tensors: name → numpy array
    tensors = {}

    def add(name, t):
        if isinstance(t, torch.Tensor):
            t = t.detach().float().numpy()
        tensors[name] = t.astype(np.float32)

    sd = model.state_dict()

    # CNN feature extractor
    n_conv = len(conv_kernel)
    for i in range(n_conv):
        p = f"wav2vec2.feature_extractor.conv_layers.{i}"
        op = f"feature_extractor.conv_layers.{i}"
        add(f"{op}.conv.weight", sd[f"{p}.conv.weight"])
        add(f"{op}.conv.bias",   sd[f"{p}.conv.bias"])
        # Layer norm or group norm
        ln_w_key = f"{p}.layer_norm.weight"
        ln_b_key = f"{p}.layer_norm.bias"
        if ln_w_key in sd:
            add(f"{op}.layer_norm.weight", sd[ln_w_key])
            add(f"{op}.layer_norm.bias",   sd[ln_b_key])

    # Feature projection
    add("feature_projection.layer_norm.weight", sd["wav2vec2.feature_projection.layer_norm.weight"])
    add("feature_projection.layer_norm.bias",   sd["wav2vec2.feature_projection.layer_norm.bias"])
    add("feature_projection.projection.weight", sd["wav2vec2.feature_projection.projection.weight"])
    add("feature_projection.projection.bias",   sd["wav2vec2.feature_projection.projection.bias"])

    # Positional conv (weight_norm → materialise)
    wv_key = "wav2vec2.encoder.pos_conv_embed.conv.weight_v"
    wg_key = "wav2vec2.encoder.pos_conv_embed.conv.weight_g"
    if wv_key in sd:
        w = compute_weight_norm(sd[wv_key].numpy(), sd[wg_key].numpy())
    else:
        w = sd["wav2vec2.encoder.pos_conv_embed.conv.weight"].numpy()
    add("encoder.pos_conv_embed.conv.weight", w)
    add("encoder.pos_conv_embed.conv.bias",   sd["wav2vec2.encoder.pos_conv_embed.conv.bias"])

    # Encoder layer norm
    add("encoder.layer_norm.weight", sd["wav2vec2.encoder.layer_norm.weight"])
    add("encoder.layer_norm.bias",   sd["wav2vec2.encoder.layer_norm.bias"])

    # Transformer layers
    n_layers = cfg.num_hidden_layers
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

    use_f16 = (dtype == "f16")
    gguf_f32 = 0
    gguf_f16 = 1

    # ── Write GGUF ──────────────────────────────────────────────────────────
    print(f"Writing {out_path} ({'f16' if use_f16 else 'f32'}) ...")

    # First pass: compute offsets
    tensor_data = {}
    offset = 0
    ALIGN = 32
    for name, arr in tensors.items():
        if use_f16:
            data = arr.astype(np.float16).tobytes()
        else:
            data = arr.astype(np.float32).tobytes()
        tensor_data[name] = data
        # pad to alignment
        pad = (ALIGN - (len(data) % ALIGN)) % ALIGN
        offset += len(data) + pad

    n_kv = 13  # number of KV entries below
    n_tensors = len(tensors)

    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", 0x46554747))  # magic GGUF
        f.write(struct.pack("<I", 3))            # version
        f.write(struct.pack("<Q", n_tensors))
        # KV count — write placeholder, fill later
        kv_count_pos = f.tell()
        f.write(struct.pack("<Q", 0))

        kv_written = 0

        def kv_u32(key, val):
            nonlocal kv_written
            write_kv_scalar(f, key, val, 4)
            kv_written += 1

        def kv_str(key, val):
            nonlocal kv_written
            write_kv_scalar(f, key, val, 8)
            kv_written += 1

        kv_u32("wav2vec2.n_conv_layers",     n_conv)
        kv_u32("wav2vec2.conv_dim",          cfg.conv_dim[0] if hasattr(cfg.conv_dim,'__len__') else cfg.conv_dim)
        kv_u32("wav2vec2.n_encoder_layers",  n_layers)
        kv_u32("wav2vec2.n_heads",           cfg.num_attention_heads)
        kv_u32("wav2vec2.hidden_size",       cfg.hidden_size)
        kv_u32("wav2vec2.intermediate_size", cfg.intermediate_size)
        kv_u32("wav2vec2.vocab_size",        len(vocab))
        kv_u32("wav2vec2.pad_token_id",      pad_id)
        kv_u32("wav2vec2.pos_conv_kernel",   pos_conv_kernel)
        kv_u32("wav2vec2.pos_conv_groups",   pos_conv_groups)
        kv_str("wav2vec2.feat_extract_norm", feat_norm)
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
        tensor_offsets = {}
        for name, arr in tensors.items():
            data   = tensor_data[name]
            ttype  = gguf_f16 if use_f16 else gguf_f32
            shape  = list(reversed(arr.shape))  # GGUF stores innermost first
            write_str(f, name)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", ttype))
            tensor_offsets[name] = cur_offset
            f.write(struct.pack("<Q", cur_offset))
            pad = (ALIGN - (len(data) % ALIGN)) % ALIGN
            cur_offset += len(data) + pad

        # Align to 32 bytes before data
        pos = f.tell()
        pad = (ALIGN - (pos % ALIGN)) % ALIGN
        f.write(b"\x00" * pad)

        # Tensor data
        for name, arr in tensors.items():
            data = tensor_data[name]
            f.write(data)
            pad = (ALIGN - (len(data) % ALIGN)) % ALIGN
            if pad: f.write(b"\x00" * pad)

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Done: {out_path} ({size_mb:.1f} MB, {n_tensors} tensors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model",  help="HuggingFace model ID or local path")
    parser.add_argument("output", help="Output .gguf file")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    args = parser.parse_args()
    convert(args.model, args.output, args.dtype)
