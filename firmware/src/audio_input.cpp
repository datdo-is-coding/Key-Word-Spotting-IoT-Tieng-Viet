#include "audio_input.h"
#include "kws_config.h"

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int s_mic_channel = 0; /* 0 for Left (L/R=GND), 1 for Right (L/R=VDD) */
static float s_ambient_rms = 0.005f;
static float s_gate_threshold = 0.010f;
static float s_gate_env = 0.0f;
static float s_gate_gain = 0.0f;
static int32_t s_raw_stereo[STEP_SAMPLES * 2];

/* ─── Biquad Filter Implementation ────────────────────────────────────── */

typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float x1, x2;
    float y1, y2;
} biquad_t;

static inline void biquad_init(biquad_t *f, float b0, float b1, float b2, float a1, float a2) {
    f->b0 = b0; f->b1 = b1; f->b2 = b2;
    f->a1 = a1; f->a2 = a2;
    f->x1 = 0.0f; f->x2 = 0.0f; f->y1 = 0.0f; f->y2 = 0.0f;
}

static inline float biquad_step(biquad_t *f, float in) {
    float out = f->b0 * in + f->b1 * f->x1 + f->b2 * f->x2 - f->a1 * f->y1 - f->a2 * f->y2;
    f->x2 = f->x1;
    f->x1 = in;
    f->y2 = f->y1;
    f->y1 = out;
    return out;
}

/* 4th-Order Filters */
static biquad_t s_hpf1, s_hpf2;
static biquad_t s_lpf1, s_lpf2;

/* ─── Audio Input Initialization ─────────────────────────────────────── */

esp_err_t audio_input_init(void) {
    // 1. Cấu hình I2S Master RX Stereo 32-bit để tự động dò kênh Left/Right của INMP441
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT, // Đọc Stereo 2 slot
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = MIC_I2S_GPIO_BCLK,
        .ws_io_num = MIC_I2S_GPIO_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = MIC_I2S_GPIO_DIN
    };

    esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[ERROR] Cài đặt I2S driver thất bại: %d\n", err);
        return err;
    }

    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[ERROR] Cài đặt I2S pin thất bại: %d\n", err);
        return err;
    }

    i2s_zero_dma_buffer(I2S_PORT);

    /* Khởi tạo 4th-Order Linkwitz-Riley HPF @ 180Hz (Fs=16kHz) lọc triệt để DC bias */
    biquad_init(&s_hpf1, 0.9512454f, -1.9024908f, 0.9512454f, -1.9001124f, 0.9048693f);
    biquad_init(&s_hpf2, 0.9512454f, -1.9024908f, 0.9512454f, -1.9001124f, 0.9048693f);

    /* Khởi tạo 4th-Order Linkwitz-Riley LPF @ 3400Hz (Fs=16kHz) lọc nhiễu cao tần */
    biquad_init(&s_lpf1, 0.2271180f, 0.4542359f, 0.2271180f, -0.2766646f, 0.1851365f);
    biquad_init(&s_lpf2, 0.2271180f, 0.4542359f, 0.2271180f, -0.2766646f, 0.1851365f);

    /* Xả đệm khởi động */
    delay(100);
    audio_input_flush_buffers();

    /* Calibration: Tự động phát hiện khe micro (Left vs Right) & hiệu chuẩn ngưỡng ồn nền */
    int64_t l_sum = 0, r_sum = 0;
    float sum_filtered_rms = 0.0f;
    int valid_frames = 0;

    for (int k = 0; k < 6; k++) {
        size_t bytes_read = 0;
        i2s_read(I2S_PORT, (char*)s_raw_stereo, sizeof(s_raw_stereo), &bytes_read, pdMS_TO_TICKS(50));
        if (bytes_read > 0) {
            int mono_cnt = (bytes_read / sizeof(int32_t)) / 2;
            float sum_sq = 0.0f;
            for (int i = 0; i < mono_cnt; i++) {
                int32_t left_s  = s_raw_stereo[i * 2];
                int32_t right_s = s_raw_stereo[i * 2 + 1];
                l_sum += (int64_t)abs(left_s >> 16);
                r_sum += (int64_t)abs(right_s >> 16);

                float s = (float)(left_s >> 16) / 32768.0f;
                s = biquad_step(&s_hpf1, s);
                s = biquad_step(&s_hpf2, s);
                s = biquad_step(&s_lpf1, s);
                s = biquad_step(&s_lpf2, s);
                sum_sq += s * s;
            }
            if (mono_cnt > 0) {
                sum_filtered_rms += sqrtf(sum_sq / (float)mono_cnt);
                valid_frames++;
            }
        }
    }

    s_mic_channel = (r_sum > l_sum * 2) ? 1 : 0;
    if (valid_frames > 0) {
        s_ambient_rms = sum_filtered_rms / (float)valid_frames;
    }
    s_gate_threshold = s_ambient_rms * 1.8f;
    if (s_gate_threshold < 0.003f) s_gate_threshold = 0.003f;
    if (s_gate_threshold > 0.035f) s_gate_threshold = 0.035f;

    Serial.printf("[AUDIO_IN] ✅ Microphone sẵn sàng | Kênh: %s | Nhiễu nền RMS: %.4f | Ngưỡng Gate: %.4f\n",
                  s_mic_channel == 0 ? "LEFT (L/R=GND)" : "RIGHT (L/R=VDD)",
                  s_ambient_rms, s_gate_threshold);

    return ESP_OK;
}

void audio_input_flush_buffers(void) {
    static int32_t dummy[512];
    size_t br = 0;
    for (int k = 0; k < 4; k++) {
        i2s_read(I2S_PORT, (char*)dummy, sizeof(dummy), &br, pdMS_TO_TICKS(30));
    }
}

float audio_input_get_ambient_rms(void) {
    return s_ambient_rms;
}

esp_err_t audio_input_read_clean(float *out_mono, int samples, float *out_rms) {
    size_t bytes_to_read = samples * 2 * sizeof(int32_t);
    if (bytes_to_read > sizeof(s_raw_stereo)) bytes_to_read = sizeof(s_raw_stereo);

    size_t bytes_read = 0;
    esp_err_t ret = i2s_read(I2S_PORT, (char*)s_raw_stereo, bytes_to_read, &bytes_read, pdMS_TO_TICKS(200));
    if (ret != ESP_OK || bytes_read == 0) {
        return ESP_FAIL;
    }

    int read_mono = (bytes_read / sizeof(int32_t)) / 2;
    if (read_mono > samples) read_mono = samples;

    float sum_sq = 0.0f;
    for (int i = 0; i < read_mono; i++) {
        /* Trích xuất từ kênh micro đã nhận diện */
        int32_t raw_32 = s_raw_stereo[i * 2 + s_mic_channel];
        float in_sample = (float)(raw_32 >> 16) / 32768.0f;

        /* 4th-Order Bandpass: 180Hz HPF + 3400Hz LPF */
        float s = biquad_step(&s_hpf1, in_sample);
        s = biquad_step(&s_hpf2, s);
        s = biquad_step(&s_lpf1, s);
        s = biquad_step(&s_lpf2, s);

        /* Envelope Follower */
        float abs_s = fabsf(s);
        if (abs_s > s_gate_env) {
            s_gate_env += 0.04f * (abs_s - s_gate_env);   /* Tấn công nhanh ~1.5ms */
        } else {
            s_gate_env += 0.001f * (abs_s - s_gate_env);  /* Nhả mượt mà ~60ms */
        }

        /* Downward Expander (Noise Gate) */
        float target_gain = 0.0f;
        if (s_gate_env > s_gate_threshold * 1.5f) {
            target_gain = 1.0f;
        } else if (s_gate_env > s_gate_threshold) {
            target_gain = (s_gate_env - s_gate_threshold) / (0.5f * s_gate_threshold);
        } else {
            target_gain = 0.0f;
        }
        s_gate_gain += 0.02f * (target_gain - s_gate_gain);

        float clean = s * s_gate_gain;
        if (clean > 1.0f) clean = 1.0f;
        if (clean < -1.0f) clean = -1.0f;

        out_mono[i] = clean;
        sum_sq += clean * clean;
    }

    if (out_rms) {
        *out_rms = sqrtf(sum_sq / (float)read_mono);
    }

    return ESP_OK;
}

bool audio_input_is_speech(float frame_rms) {
    float speech_thresh = s_ambient_rms * 2.2f;
    if (speech_thresh < 0.012f) {
        speech_thresh = 0.012f;
    }
    return (frame_rms >= speech_thresh);
}
