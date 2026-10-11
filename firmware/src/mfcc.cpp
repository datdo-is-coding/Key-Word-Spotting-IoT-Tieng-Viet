#include "mfcc.h"
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

MFCCExtractor::MFCCExtractor() {
}

MFCCExtractor::~MFCCExtractor() {
}

bool MFCCExtractor::init() {
    init_hamming();
    init_mel_filterbank();
    init_dct();
    init_lifter();
    return true;
}

void MFCCExtractor::init_hamming() {
    for (int i = 0; i < FRAME_LEN_SAMPLES; i++) {
        hamming_window[i] = 0.54f - 0.46f * cosf((2.0f * (float)M_PI * (float)i) / (float)(FRAME_LEN_SAMPLES - 1));
    }
}

static inline float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void MFCCExtractor::init_mel_filterbank() {
    memset(mel_filters, 0, sizeof(mel_filters));
    float low_mel = hz_to_mel(300.0f);
    float high_mel = hz_to_mel(8000.0f);

    float mel_points[NUM_MEL_FILTERS + 2];
    int bin_points[NUM_MEL_FILTERS + 2];

    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        mel_points[i] = low_mel + (float)i * (high_mel - low_mel) / (float)(NUM_MEL_FILTERS + 1);
        float hz = mel_to_hz(mel_points[i]);
        bin_points[i] = (int)floorf((float)(FFT_SIZE + 1) * hz / (float)SAMPLE_RATE);
    }

    for (int m = 1; m <= NUM_MEL_FILTERS; m++) {
        int f_minus = bin_points[m - 1];
        int f_curr = bin_points[m];
        int f_plus = bin_points[m + 1];

        for (int k = f_minus; k < f_curr; k++) {
            if (f_curr != f_minus) {
                mel_filters[m - 1][k] = (float)(k - f_minus) / (float)(f_curr - f_minus);
            }
        }
        for (int k = f_curr; k < f_plus; k++) {
            if (f_plus != f_curr) {
                mel_filters[m - 1][k] = (float)(f_plus - k) / (float)(f_plus - f_curr);
            }
        }
    }
}

void MFCCExtractor::init_dct() {
    float factor0 = sqrtf(1.0f / (float)NUM_MEL_FILTERS);
    float factor = sqrtf(2.0f / (float)NUM_MEL_FILTERS);

    for (int i = 0; i < NUM_MFCC_COEFFS; i++) {
        for (int j = 0; j < NUM_MEL_FILTERS; j++) {
            float angle = (float)M_PI * (float)i * ((float)j + 0.5f) / (float)NUM_MEL_FILTERS;
            dct_matrix[i][j] = (i == 0 ? factor0 : factor) * cosf(angle);
        }
    }
}

void MFCCExtractor::init_lifter() {
    float cep_lifter = 22.0f;
    for (int n = 0; n < NUM_MFCC_COEFFS; n++) {
        lifter[n] = 1.0f + (cep_lifter / 2.0f) * sinf((float)M_PI * (float)n / cep_lifter);
    }
}

// Thuật toán FFT Cooley-Tukey Radix-2 512-điểm
void MFCCExtractor::rfft(const float* input, float* real_out, float* imag_out) {
    for (int i = 0; i < FFT_SIZE; i++) {
        real_out[i] = (i < FRAME_LEN_SAMPLES) ? input[i] : 0.0f;
        imag_out[i] = 0.0f;
    }

    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < FFT_SIZE - 1; i++) {
        if (i < j) {
            float temp_r = real_out[i];
            float temp_i = imag_out[i];
            real_out[i] = real_out[j];
            imag_out[i] = imag_out[j];
            real_out[j] = temp_r;
            imag_out[j] = temp_i;
        }
        int k = FFT_SIZE >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    // Butterfly computations
    for (int len = 2; len <= FFT_SIZE; len <<= 1) {
        float angle = -2.0f * (float)M_PI / (float)len;
        float wlen_r = cosf(angle);
        float wlen_i = sinf(angle);

        for (int i = 0; i < FFT_SIZE; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                float u_r = real_out[i + k];
                float u_i = imag_out[i + k];
                float v_r = real_out[i + k + len / 2] * w_r - imag_out[i + k + len / 2] * w_i;
                float v_i = real_out[i + k + len / 2] * w_i + imag_out[i + k + len / 2] * w_r;

                real_out[i + k] = u_r + v_r;
                imag_out[i + k] = u_i + v_i;
                real_out[i + k + len / 2] = u_r - v_r;
                imag_out[i + k + len / 2] = u_i - v_i;

                float next_w_r = w_r * wlen_r - w_i * wlen_i;
                float next_w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = next_w_r;
                w_i = next_w_i;
            }
        }
    }
}

void MFCCExtractor::extract(const float* audio_in, float* mfcc_out) {
    // 1. Pre-emphasis: y[t] = x[t] - 0.97 * x[t-1]
    static float emphasized[TOTAL_SAMPLES];
    emphasized[0] = audio_in[0];
    for (int i = 1; i < TOTAL_SAMPLES; i++) {
        emphasized[i] = audio_in[i] - 0.97f * audio_in[i - 1];
    }

    float frame_buf[FRAME_LEN_SAMPLES];
    float real_fft[FFT_SIZE];
    float imag_fft[FFT_SIZE];
    float power_spectrum[FFT_SIZE / 2 + 1];
    float filter_energies[NUM_MEL_FILTERS];

    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        int start_idx = frame * FRAME_STEP_SAMPLES;

        // Windowing
        for (int i = 0; i < FRAME_LEN_SAMPLES; i++) {
            frame_buf[i] = emphasized[start_idx + i] * hamming_window[i];
        }

        // RFFT
        rfft(frame_buf, real_fft, imag_fft);

        // Power spectrum: (|real|^2 + |imag|^2) / FFT_SIZE
        for (int k = 0; k <= FFT_SIZE / 2; k++) {
            power_spectrum[k] = (real_fft[k] * real_fft[k] + imag_fft[k] * imag_fft[k]) / (float)FFT_SIZE;
        }

        // Mel Filterbank energies
        for (int m = 0; m < NUM_MEL_FILTERS; m++) {
            float energy = 0.0f;
            for (int k = 0; k <= FFT_SIZE / 2; k++) {
                energy += power_spectrum[k] * mel_filters[m][k];
            }
            if (energy < 1e-10f) energy = 1e-10f;
            filter_energies[m] = logf(energy);
        }

        // DCT-II + Liftering + Z-Score Normalization
        int out_offset = frame * NUM_MFCC_COEFFS;
        for (int c = 0; c < NUM_MFCC_COEFFS; c++) {
            float coeff = 0.0f;
            for (int m = 0; m < NUM_MEL_FILTERS; m++) {
                coeff += dct_matrix[c][m] * filter_energies[m];
            }
            coeff *= lifter[c];
            // Chuẩn hóa theo mean và std của tập train
            mfcc_out[out_offset + c] = (coeff - NORM_MEAN) / NORM_STD;
        }
    }
}
