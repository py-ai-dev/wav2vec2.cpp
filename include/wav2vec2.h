#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct wav2vec2_context;

// Inference result — caller must free() transcript
struct wav2vec2_result {
    char   * transcript;
    int      n_frames;      // number of feature frames processed
};

struct wav2vec2_params {
    int  n_threads;         // default: 4
    bool verbose;           // print timing info
};

struct wav2vec2_params wav2vec2_default_params(void);

// Load model from a .gguf file produced by scripts/convert_to_gguf.py
struct wav2vec2_context * wav2vec2_init(const char * path_model);
void                      wav2vec2_free(struct wav2vec2_context * ctx);

// Transcribe raw PCM samples: 16 kHz, mono, float32 normalised to [-1, 1]
// Returns heap-allocated result; caller must free result->transcript then free(result)
struct wav2vec2_result * wav2vec2_transcribe(
    struct wav2vec2_context * ctx,
    struct wav2vec2_params    params,
    const float             * samples,
    int                       n_samples);

#ifdef __cplusplus
}
#endif
