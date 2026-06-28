//  wav2vec2-cli — transcribe a WAV file using a .gguf wav2vec2 model
//
//  Usage: wav2vec2-cli -m model.gguf -f audio.wav [-t N] [-b N] [--format txt|json|srt|vtt] [-w] [-v]

#include "wav2vec2.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

// ── WAV reader ───────────────────────────────────────────────────────────────
// PCM 16-bit or 32-bit float, mono or stereo, any sample rate.
// Downmixes to mono and resamples (linear interpolation) to 16 kHz.

static std::vector<float> load_wav(const char * path, int target_sr = 16000) {
    FILE * f = fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("Cannot open WAV: ") + path);

    auto rd4 = [&]{ uint32_t v=0; fread(&v,4,1,f); return v; };
    auto rd2 = [&]{ uint16_t v=0; fread(&v,2,1,f); return v; };

    uint32_t riff = rd4();
    if (riff != 0x46464952u) throw std::runtime_error("Not a RIFF file");
    rd4();
    uint32_t wave = rd4();
    if (wave != 0x45564157u) throw std::runtime_error("Not a WAVE file");

    uint16_t audio_fmt=0, n_ch=0, bits=0;
    uint32_t sr=0, data_size=0;
    bool found_fmt=false, found_data=false;

    while (!feof(f)) {
        uint32_t chunk_id   = rd4();
        uint32_t chunk_size = rd4();
        if (chunk_id == 0x20746d66u) {
            audio_fmt = rd2(); n_ch = rd2(); sr = rd4();
            rd4(); rd2();
            bits = rd2();
            if (chunk_size > 16) fseek(f, (long)(chunk_size - 16), SEEK_CUR);
            found_fmt = true;
        } else if (chunk_id == 0x61746164u) {
            data_size = chunk_size; found_data = true; break;
        } else {
            fseek(f, (long)chunk_size, SEEK_CUR);
        }
    }
    if (!found_fmt || !found_data)  throw std::runtime_error("Malformed WAV");
    if (audio_fmt != 1 && audio_fmt != 3) throw std::runtime_error("Only PCM/float WAV");

    uint32_t n_samples = data_size / (n_ch * (bits / 8));
    std::vector<float> mono(n_samples);
    for (uint32_t i = 0; i < n_samples; i++) {
        float sum = 0.f;
        for (uint16_t c = 0; c < n_ch; c++) {
            if      (audio_fmt == 1 && bits == 16) { int16_t s; fread(&s,2,1,f); sum += s/32768.f; }
            else if (audio_fmt == 1 && bits == 32) { int32_t s; fread(&s,4,1,f); sum += s/2147483648.f; }
            else if (audio_fmt == 3 && bits == 32) { float   s; fread(&s,4,1,f); sum += s; }
            else throw std::runtime_error("Unsupported bit depth");
        }
        mono[i] = sum / n_ch;
    }
    fclose(f);

    if ((int)sr == target_sr) return mono;

    // Linear interpolation resampling
    uint32_t out_n = (uint32_t)((uint64_t)n_samples * target_sr / sr);
    std::vector<float> out(out_n);
    for (uint32_t i = 0; i < out_n; i++) {
        double pos  = (double)i * sr / target_sr;
        uint32_t lo = (uint32_t)pos;
        uint32_t hi = std::min(lo + 1u, n_samples - 1u);
        float frac  = (float)(pos - lo);
        out[i] = mono[lo] * (1.f - frac) + mono[hi] * frac;
    }
    return out;
}

// ── Output formatters ────────────────────────────────────────────────────────

static std::string srt_time(float s) {
    int h   = (int)(s / 3600);
    int m   = (int)(s / 60) % 60;
    int sec = (int)s % 60;
    int ms  = (int)(s * 1000) % 1000;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d,%03d", h, m, sec, ms);
    return buf;
}

static std::string vtt_time(float s) {
    int m   = (int)(s / 60);
    int sec = (int)s % 60;
    int ms  = (int)(s * 1000) % 1000;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d.%03d", m, sec, ms);
    return buf;
}

// Minimal JSON string escaper
static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else                out += c;
    }
    return out;
}

static void print_txt(const wav2vec2_result * r) {
    printf("%s\n", r->transcript);
}

static void print_json(const wav2vec2_result * r) {
    float dur = r->n_frames * (320.f / 16000.f);
    printf("{\"transcript\":\"%s\",\"duration\":%.3f,\"n_frames\":%d",
           json_escape(r->transcript).c_str(), dur, r->n_frames);
    if (r->n_words > 0) {
        printf(",\"words\":[");
        for (int i = 0; i < r->n_words; i++) {
            if (i) printf(",");
            printf("{\"word\":\"%s\",\"start\":%.3f,\"end\":%.3f}",
                   json_escape(r->words[i].text).c_str(),
                   r->words[i].t_start, r->words[i].t_end);
        }
        printf("]");
    }
    printf("}\n");
}

static void print_srt(const wav2vec2_result * r) {
    float dur = r->n_frames * (320.f / 16000.f);
    if (r->n_words > 0) {
        // One subtitle block per word
        for (int i = 0; i < r->n_words; i++) {
            printf("%d\n%s --> %s\n%s\n\n",
                   i + 1,
                   srt_time(r->words[i].t_start).c_str(),
                   srt_time(r->words[i].t_end).c_str(),
                   r->words[i].text);
        }
    } else {
        // Single block spanning the whole file
        printf("1\n%s --> %s\n%s\n\n",
               srt_time(0.f).c_str(), srt_time(dur).c_str(), r->transcript);
    }
}

static void print_vtt(const wav2vec2_result * r) {
    float dur = r->n_frames * (320.f / 16000.f);
    printf("WEBVTT\n\n");
    if (r->n_words > 0) {
        for (int i = 0; i < r->n_words; i++) {
            printf("%s --> %s\n%s\n\n",
                   vtt_time(r->words[i].t_start).c_str(),
                   vtt_time(r->words[i].t_end).c_str(),
                   r->words[i].text);
        }
    } else {
        printf("%s --> %s\n%s\n\n",
               vtt_time(0.f).c_str(), vtt_time(dur).c_str(), r->transcript);
    }
}

// ── CLI ──────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * audio_path = nullptr;
    int n_threads  = 4;
    int beam_size  = 1;
    bool verbose   = false;
    bool do_words  = false;
    std::string fmt = "txt";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m") && i+1<argc)       model_path = argv[++i];
        else if (!strcmp(argv[i], "-f") && i+1<argc)       audio_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i+1<argc)       n_threads  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i+1<argc)       beam_size  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--format") && i+1<argc) fmt        = argv[++i];
        else if (!strcmp(argv[i], "-v"))                   verbose    = true;
        else if (!strcmp(argv[i], "-w") || !strcmp(argv[i], "--words")) do_words = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                "Usage: %s -m model.gguf -f audio.wav [options]\n\n"
                "  -m  PATH      GGUF model file\n"
                "  -f  PATH      input WAV (any rate/channels; resampled to 16 kHz mono)\n"
                "  -t  N         threads (default: 4)\n"
                "  -b  N         CTC beam width: 1=greedy, >1=beam search (default: 1)\n"
                "  --format FMT  output format: txt (default), json, srt, vtt\n"
                "  -w/--words    include word-level timestamps (json/srt/vtt auto-enable)\n"
                "  -v            verbose timing on stderr\n", argv[0]);
            return 0;
        }
    }

    if (!model_path || !audio_path) {
        fprintf(stderr, "Usage: %s -m model.gguf -f audio.wav\n", argv[0]);
        return 1;
    }

    // srt/vtt always need word timestamps
    if (fmt == "srt" || fmt == "vtt" || fmt == "json") do_words = true;

    try {
        if (verbose) fprintf(stderr, "Loading model: %s\n", model_path);
        auto * ctx = wav2vec2_init(model_path);
        if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

        if (verbose) fprintf(stderr, "Loading audio: %s\n", audio_path);
        auto samples = load_wav(audio_path);
        if (verbose) fprintf(stderr, "Audio: %zu samples (%.2f s)\n",
                             samples.size(), samples.size() / 16000.0);

        auto params          = wav2vec2_default_params();
        params.n_threads     = n_threads;
        params.beam_size     = beam_size;
        params.verbose       = verbose;
        params.word_timestamps = do_words;

        auto * res = wav2vec2_transcribe(ctx, params, samples.data(), (int)samples.size());

        if      (fmt == "json") print_json(res);
        else if (fmt == "srt")  print_srt(res);
        else if (fmt == "vtt")  print_vtt(res);
        else                    print_txt(res);

        wav2vec2_result_free(res);
        wav2vec2_free(ctx);
    } catch (const std::exception & e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
