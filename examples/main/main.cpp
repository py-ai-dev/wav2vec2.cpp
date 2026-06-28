//  wav2vec2-cli — transcribe a WAV file using a .gguf wav2vec2 model
//
//  Usage: wav2vec2-cli -m model.gguf -f audio.wav [-t threads] [-b beam] [-v]

#include "wav2vec2.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

// ── Minimal WAV reader ───────────────────────────────────────────────────────
// Supports: PCM 16-bit or 32-bit float, mono or stereo, any sample rate.
// Downmixes to mono and resamples (nearest-neighbour) to 16 kHz.

static std::vector<float> load_wav(const char * path, int target_sr = 16000) {
    FILE * f = fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("Cannot open WAV: ") + path);

    auto rd4 = [&]{ uint32_t v=0; fread(&v,4,1,f); return v; };
    auto rd2 = [&]{ uint16_t v=0; fread(&v,2,1,f); return v; };

    // RIFF header
    uint32_t riff = rd4();
    if (riff != 0x46464952u) throw std::runtime_error("Not a RIFF file");
    rd4(); // file size
    uint32_t wave = rd4();
    if (wave != 0x45564157u) throw std::runtime_error("Not a WAVE file");

    uint16_t audio_fmt=0, n_ch=0, bits=0;
    uint32_t sr=0, data_size=0;
    bool found_fmt=false, found_data=false;

    while (!feof(f)) {
        uint32_t chunk_id   = rd4();
        uint32_t chunk_size = rd4();
        if (chunk_id == 0x20746d66u) { // "fmt "
            audio_fmt = rd2();
            n_ch      = rd2();
            sr        = rd4();
            rd4(); rd2(); // byte rate, block align
            bits      = rd2();
            if (chunk_size > 16) fseek(f, (long)(chunk_size - 16), SEEK_CUR);
            found_fmt = true;
        } else if (chunk_id == 0x61746164u) { // "data"
            data_size = chunk_size;
            found_data = true;
            break;
        } else {
            fseek(f, (long)chunk_size, SEEK_CUR);
        }
    }
    if (!found_fmt || !found_data)
        throw std::runtime_error("Malformed WAV");
    if (audio_fmt != 1 && audio_fmt != 3)
        throw std::runtime_error("Only PCM (1) or IEEE float (3) WAV supported");

    uint32_t n_samples = data_size / (n_ch * (bits / 8));
    std::vector<float> mono(n_samples);

    for (uint32_t i = 0; i < n_samples; i++) {
        float sum = 0.f;
        for (uint16_t c = 0; c < n_ch; c++) {
            if (audio_fmt == 1 && bits == 16) {
                int16_t s; fread(&s, 2, 1, f); sum += s / 32768.f;
            } else if (audio_fmt == 1 && bits == 32) {
                int32_t s; fread(&s, 4, 1, f); sum += s / 2147483648.f;
            } else if (audio_fmt == 3 && bits == 32) {
                float s; fread(&s, 4, 1, f); sum += s;
            } else {
                throw std::runtime_error("Unsupported bit depth");
            }
        }
        mono[i] = sum / n_ch;
    }
    fclose(f);

    // Nearest-neighbour resample to target_sr
    if ((int)sr == target_sr) return mono;
    uint32_t out_n = (uint32_t)((uint64_t)n_samples * target_sr / sr);
    std::vector<float> out(out_n);
    for (uint32_t i = 0; i < out_n; i++)
        out[i] = mono[(uint64_t)i * sr / target_sr];
    return out;
}

// ── CLI ──────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * audio_path = nullptr;
    int n_threads = 4;
    int beam_size = 1;
    bool verbose  = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i+1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-f") && i+1 < argc) audio_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i+1 < argc) beam_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr, "Usage: %s -m model.gguf -f audio.wav [-t threads] [-b beam] [-v]\n", argv[0]);
            return 0;
        }
    }

    if (!model_path || !audio_path) {
        fprintf(stderr, "Usage: %s -m model.gguf -f audio.wav\n", argv[0]);
        return 1;
    }

    try {
        if (verbose) fprintf(stderr, "Loading model: %s\n", model_path);
        auto * ctx = wav2vec2_init(model_path);
        if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

        if (verbose) fprintf(stderr, "Loading audio: %s\n", audio_path);
        auto samples = load_wav(audio_path);
        if (verbose) fprintf(stderr, "Audio: %zu samples (%.2f s)\n",
                             samples.size(), samples.size() / 16000.0);

        auto params = wav2vec2_default_params();
        params.n_threads = n_threads;
        params.beam_size = beam_size;
        params.verbose   = verbose;

        auto * res = wav2vec2_transcribe(ctx, params, samples.data(), (int)samples.size());
        printf("%s\n", res->transcript);

        free(res->transcript);
        free(res);
        wav2vec2_free(ctx);
    } catch (const std::exception & e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
