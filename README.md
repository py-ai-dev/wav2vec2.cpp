<p align="center">
  <img src="media/logo.svg" alt="wav2vec2.cpp" width="720"/>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"/></a>
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17"/>
  <img src="https://img.shields.io/badge/deps-none-brightgreen.svg" alt="No dependencies"/>
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20mac%20%7C%20windows-lightgrey.svg" alt="Cross-platform"/>
</p>

---

**Fast CPU inference for [wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) ASR models — no Python, no PyTorch, no CUDA required.**

[whisper.cpp](https://github.com/ggerganov/whisper.cpp) did this for Whisper. wav2vec2 has thousands of community fine-tuned models on HuggingFace — Telugu, Tamil, Arabic, Hindi, Swahili, Bangla, 100+ languages — and none of them had a C++ inference path until now.

---

## Features

- **Zero dependencies** — C++17 stdlib only; optional OpenBLAS for faster matmul
- **Universal** — converts any `Wav2Vec2ForCTC` model from HuggingFace to GGUF
- **F16 + F32** — half-precision storage halves model size with minimal quality loss
- **Built-in WAV reader** — no libsndfile, no miniaudio; just a file path
- **Greedy + beam search CTC** — greedy by default (`-b 1`), full prefix beam search with `-b N`
- **Multi-threaded attention** — transformer heads split across CPU cores (`-t N`)
- **BLAS-accelerated matmul** — auto-detected at configure time (OpenBLAS, MKL, Accelerate)
- **ARM + x86** — NEON (armv8.2-a+fp16) and AVX2+FMA compiler flags
- **Tested** — unit tests for every math primitive, the GGUF reader, and the beam search decoder

## Build

```bash
git clone https://github.com/py-ai-dev/wav2vec2.cpp
cd wav2vec2.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Run tests:
```bash
cd build && ctest --output-on-failure
```

**Requirements:** C++17 compiler, CMake ≥ 3.14. OpenBLAS is auto-detected for faster matrix multiply but is not required.

## Quick Start

### 1. Convert a model

```bash
pip install transformers torch
python scripts/convert_to_gguf.py facebook/wav2vec2-base-960h model.gguf
```

Add `--dtype f32` for full precision (default is f16, ~2× smaller file).

### 2. Transcribe

```bash
./build/wav2vec2-cli -m model.gguf -f audio.wav
```

```
usage: wav2vec2-cli -m MODEL -f AUDIO [-t THREADS] [-b BEAM] [-v]

  -m  model.gguf    GGUF model file
  -f  audio.wav     input WAV (16 kHz mono recommended; stereo/other rates auto-handled)
  -t  4             number of threads (default: 4)
  -b  1             CTC beam width: 1 = greedy, 5 = beam search (default: 1)
  -v                verbose output with timing
```

## Examples

```bash
# English (360 MB base model)
python scripts/convert_to_gguf.py facebook/wav2vec2-base-960h      en.gguf
./build/wav2vec2-cli -m en.gguf -f speech.wav

# Telugu
python scripts/convert_to_gguf.py vasista22/wav2vec2-telugu-large   te.gguf
./build/wav2vec2-cli -m te.gguf -f telugu.wav

# Arabic (XLS-R fine-tune)
python scripts/convert_to_gguf.py jonatasgrosman/wav2vec2-large-xlsr-53-arabic  ar.gguf
./build/wav2vec2-cli -m ar.gguf -f arabic.wav

# Any other language — same pattern
python scripts/convert_to_gguf.py <any-wav2vec2-ctc-model> out.gguf
./build/wav2vec2-cli -m out.gguf -f audio.wav
```

## Architecture

```
raw audio (16 kHz float32)
  ↓
CNN Feature Extractor     7 conv layers, total stride 320 → ~49 frames/sec
  ↓
Feature Projection        linear + layer norm  [T × conv_dim → T × hidden]
  ↓
Positional Conv Embed     grouped Conv1D, adds position information
  ↓
Transformer Encoder       12–24 layers, full bidirectional self-attention
  ↓
CTC Head                  linear → argmax over vocab → remove dups/blanks
  ↓
transcript
```

## Supported Models

Any `Wav2Vec2ForCTC` checkpoint with:

| Config field | Supported values |
|---|---|
| `feat_extract_norm` | `"group"` (base) or `"layer"` (large/XLS-R) |
| `feat_extract_activation` | `"gelu"` |
| Architecture | standard 7-layer CNN + transformer |

This covers **wav2vec2-base**, **wav2vec2-large**, **XLS-R-300M**, **XLS-R-1B**, and the vast majority of community fine-tunes.

## Performance

Tested on a 20-core ARM (Cortex-X925 + A725), 120 GB RAM:

| Model | Size | 10s audio | RTF |
|---|---|---|---|
| wav2vec2-base-960h | 360 MB | ~0.8 s | 0.08 |
| wav2vec2-large-xlsr | 1.18 GB | ~3.2 s | 0.32 |

RTF < 1.0 = faster than real time. Both models run comfortably on CPU with no GPU.

## Repository Structure

```
wav2vec2.cpp/
├── include/wav2vec2.h          public C API
├── src/
│   ├── ops.h                   math primitives (gelu, layer_norm, conv1d, …)
│   ├── gguf.h                  minimal GGUF reader
│   └── wav2vec2.cpp            model loading + forward pass
├── examples/main/main.cpp      CLI tool
├── tests/
│   ├── test_ops.cpp            unit tests — all math ops
│   └── test_gguf.cpp           unit tests — GGUF reader
├── scripts/convert_to_gguf.py  HuggingFace → GGUF converter
└── CMakeLists.txt
```

## Roadmap

- [ ] Language model rescoring (KenLM)
- [ ] Quantisation (Q8, Q4) via ggml integration
- [ ] Python bindings
- [ ] WASM / browser support
- [ ] Android / iOS examples
- [ ] Batch inference
- [ ] Metal / CUDA backend

## Contributing

PRs welcome. Please add or update tests for any changed logic. Run `ctest` before submitting.

## License

MIT — see [LICENSE](LICENSE).

Produced by [liodon-ai](https://huggingface.co/liodon-ai).
