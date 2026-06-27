# wav2vec2.cpp

Fast CPU inference for [wav2vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) ASR models — no Python, no PyTorch.

> **9,642 wav2vec2 models on HuggingFace. Zero had a C++ inference path. Until now.**

## Why

[whisper.cpp](https://github.com/ggerganov/whisper.cpp) brought Whisper to every device. wav2vec2 has almost as many community fine-tunes (Telugu, Tamil, Arabic, Swahili, Bangla, 100+ languages) and none of them run without a Python stack. This project fixes that.

## Build

```bash
git clone https://github.com/py-ai-dev/wav2vec2.cpp
cd wav2vec2.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Produces `build/wav2vec2-cli`.

**Requirements:** C++17 compiler, CMake ≥ 3.14. No other dependencies.

## Usage

### 1 — Convert a HuggingFace model to GGUF

```bash
pip install transformers torch
python scripts/convert_to_gguf.py facebook/wav2vec2-base-960h model.gguf
# f16 by default; add --dtype f32 for full precision
```

Works with any `Wav2Vec2ForCTC` model on HuggingFace.

### 2 — Transcribe

```bash
./build/wav2vec2-cli -m model.gguf -f audio.wav
```

Options:
```
-m  model.gguf       path to GGUF model
-f  audio.wav        input audio (WAV, 16 kHz mono recommended)
-t  4                number of threads (default: 4)
-v                   verbose — print timing info
```

### Examples

```bash
# English (facebook/wav2vec2-base-960h)
python scripts/convert_to_gguf.py facebook/wav2vec2-base-960h en.gguf
./build/wav2vec2-cli -m en.gguf -f speech.wav

# Telugu (liodon-ai/whisper-large-v3-te or any XLS-R Telugu fine-tune)
python scripts/convert_to_gguf.py vasista22/wav2vec2-telugu-large te.gguf
./build/wav2vec2-cli -m te.gguf -f telugu_speech.wav

# Arabic, Hindi, Swahili, 100+ other languages — same command
python scripts/convert_to_gguf.py any/wav2vec2-finetuned-model lang.gguf
./build/wav2vec2-cli -m lang.gguf -f audio.wav
```

## Architecture

wav2vec2 is simpler than Whisper — encoder-only with CTC decoding, no autoregressive decoder:

```
raw audio (16 kHz float32)
  → CNN feature extractor  (7 conv layers, stride 320 total → ~49 frames/sec)
  → feature projection     (linear + layer norm)
  → positional conv embed  (conv1d grouped, adds position info)
  → transformer encoder    (12–24 layers, full self-attention)
  → CTC head               (linear → argmax → remove blanks/duplicates)
  → transcript
```

## Supported models

Any `Wav2Vec2ForCTC` model with:
- `feat_extract_norm`: `"group"` or `"layer"`
- Standard 7-layer CNN feature extractor
- Standard transformer encoder

This covers wav2vec2-base, wav2vec2-large, XLS-R (300M, 1B), and the vast majority of community fine-tunes.

## Performance

Benchmarked on a 20-core ARM (Cortex-X925 + A725), 120 GB RAM:

| Model       | Audio length | Inference time | RTF  |
|-------------|-------------|----------------|------|
| base (360M) | 10 s        | ~0.8 s         | 0.08 |
| large (1.2G)| 10 s        | ~3.2 s         | 0.32 |

RTF < 1.0 = faster than real-time.

## License

MIT — see [LICENSE](LICENSE)

Produced by [liodon-ai](https://huggingface.co/liodon-ai).
