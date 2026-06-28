# {repo_name}

GGUF quantizations of [{source_model}](https://huggingface.co/{source_model}) for use with [wav2vec2.cpp](https://github.com/liodon-ai/wav2vec2.cpp).

## Files

| Filename | Quant | Size | Precision |
|----------|-------|------|-----------|
| `model_f16.gguf`  | F16  | {size_f16}  | ~fp16 (near-lossless) |
| `model_q8_0.gguf` | Q8_0 | {size_q8_0} | 8-bit quantized linear layers |
| `model_q4_0.gguf` | Q4_0 | {size_q4_0} | 4-bit quantized linear layers |

## Parity vs HuggingFace Reference

Tested on {n_samples} audio samples from {test_dataset}.

| Quant | Mean CER (vs HF) | Mean WER (vs HF) | Status |
|-------|-----------------|-----------------|--------|
| F16   | {cer_f16:.2%}  | {wer_f16:.2%}  | {status_f16} |
| Q8_0  | {cer_q8_0:.2%} | {wer_q8_0:.2%} | {status_q8_0} |
| Q4_0  | {cer_q4_0:.2%} | {wer_q4_0:.2%} | {status_q4_0} |

CER/WER measured against HF model output (not ground truth) to isolate quantization error.

## Usage

### C++ CLI

```bash
# Clone and build
git clone --recursive https://github.com/liodon-ai/wav2vec2.cpp
cd wav2vec2.cpp && mkdir build && cd build && cmake .. && make -j

# Download and run
huggingface-cli download {repo_full_name} model_q8_0.gguf --local-dir .
./wav2vec2-cli -m model_q8_0.gguf -f audio.wav

# With word timestamps
./wav2vec2-cli -m model_q8_0.gguf -f audio.wav -w

# SRT subtitle output
./wav2vec2-cli -m model_q8_0.gguf -f audio.wav --format srt > output.srt
```

### Python (via subprocess)

```python
import subprocess, json
result = subprocess.run(
    ['./wav2vec2-cli', '-m', 'model_q8_0.gguf', '-f', 'audio.wav', '--format', 'json'],
    capture_output=True, text=True
)
data = json.loads(result.stdout)
print(data['transcript'])
```

## Source Model

- **Original**: [{source_model}](https://huggingface.co/{source_model})
- **Architecture**: {architecture}
- **Language**: {language}
- **Task**: Automatic Speech Recognition

## Conversion

Converted using [wav2vec2.cpp](https://github.com/liodon-ai/wav2vec2.cpp) `scripts/convert_to_gguf.py`.

```bash
python scripts/convert_to_gguf.py {source_model} model_f16.gguf --dtype f16
python scripts/convert_to_gguf.py {source_model} model_q8_0.gguf --dtype q8_0
python scripts/convert_to_gguf.py {source_model} model_q4_0.gguf --dtype q4_0
```

## License

Follows the license of the [source model]({source_license_url}).
