#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct wav2vec2_context;

// Word-level timing (populated when params.word_timestamps == true)
struct wav2vec2_word {
    char  * text;       // heap-allocated; freed by wav2vec2_result_free
    float   t_start;    // seconds
    float   t_end;      // seconds
};

// Inference result — use wav2vec2_result_free() to release
struct wav2vec2_result {
    char                  * transcript;  // full text, heap-allocated
    int                     n_frames;    // CTC feature frames (each = 20 ms)
    struct wav2vec2_word  * words;       // NULL unless params.word_timestamps
    int                     n_words;     // 0 unless params.word_timestamps
};

struct wav2vec2_params {
    int  n_threads;          // default: 4
    int  beam_size;          // 1 = greedy CTC, >1 = beam search (default: 1)
    bool verbose;            // print timing to stderr
    bool word_timestamps;    // populate result->words with per-word timing
};

struct wav2vec2_params wav2vec2_default_params(void);

// Load model from a .gguf file produced by scripts/convert_to_gguf.py
struct wav2vec2_context * wav2vec2_init(const char * path_model);
void                      wav2vec2_free(struct wav2vec2_context * ctx);

// Transcribe raw PCM: 16 kHz mono float32 in [-1, 1]
struct wav2vec2_result * wav2vec2_transcribe(
    struct wav2vec2_context * ctx,
    struct wav2vec2_params    params,
    const float             * samples,
    int                       n_samples);

// Free a result returned by wav2vec2_transcribe
void wav2vec2_result_free(struct wav2vec2_result * result);

#ifdef __cplusplus
}
#endif
