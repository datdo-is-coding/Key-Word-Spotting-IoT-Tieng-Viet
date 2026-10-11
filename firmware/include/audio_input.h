#ifndef AUDIO_INPUT_H
#define AUDIO_INPUT_H

#include <stdint.h>
#include <stdbool.h>
#include <esp_err.h>

esp_err_t audio_input_init(void);
void audio_input_flush_buffers(void);
float audio_input_get_ambient_rms(void);
esp_err_t audio_input_read_clean(float *out_mono, int samples, float *out_rms);
bool audio_input_is_speech(float frame_rms);

#endif // AUDIO_INPUT_H
