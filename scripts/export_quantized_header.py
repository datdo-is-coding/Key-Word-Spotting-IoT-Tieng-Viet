import os
import sys
import torch
import numpy as np

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_PATH = "d:/KWS/models/kws_dscnn_best.pt"
OUTPUT_HEADER = "d:/KWS/firmware/include/kws_model_quantized.h"

def quantize_per_channel(weight_tensor):
    """
    Quantize weights to signed INT8 [-127, 127] per output channel (filter).
    weight_tensor: shape (out_channels, in_channels, H, W) or (out_channels, in_channels)
    """
    dims = list(range(1, weight_tensor.ndim))
    max_val = torch.amax(torch.abs(weight_tensor), dim=dims, keepdim=True)
    scale = max_val / 127.0
    scale[scale == 0] = 1.0  # tránh chia cho 0
    w_q = torch.clamp(torch.round(weight_tensor / scale), -127, 127).to(torch.int8)
    scale_flat = scale.squeeze().cpu().numpy()
    w_q_flat = w_q.flatten().cpu().numpy()
    return w_q_flat, scale_flat

def format_int8_array(arr, per_line=16):
    lines = []
    for i in range(0, len(arr), per_line):
        chunk = arr[i:i + per_line]
        lines.append("    " + ", ".join(f"{int(x):4d}" for x in chunk) + ",")
    return "\n".join(lines)

def format_float_array(arr, per_line=8):
    lines = []
    for i in range(0, len(arr), per_line):
        chunk = arr[i:i + per_line]
        lines.append("    " + ", ".join(f"{float(x):.8e}f" for x in chunk) + ",")
    return "\n".join(lines)

def export_quantized_header():
    print(f"Loading weights from {MODEL_PATH}...")
    state = torch.load(MODEL_PATH, map_location='cpu')

    eps = 1e-5
    # 1. Conv1 + BN1
    s1 = state['bn1.weight'] / torch.sqrt(state['bn1.running_var'] + eps)
    w_c1 = state['conv1.weight'] * s1.view(-1, 1, 1, 1)
    b_c1 = (state['conv1.bias'] - state['bn1.running_mean']) * s1 + state['bn1.bias']

    # 2. DW2
    w_dw2 = state['dw2.weight']
    b_dw2 = state['dw2.bias']

    # 3. PW2 + BN2
    s2 = state['bn2.weight'] / torch.sqrt(state['bn2.running_var'] + eps)
    w_pw2 = state['pw2.weight'] * s2.view(-1, 1, 1, 1)
    b_pw2 = (state['pw2.bias'] - state['bn2.running_mean']) * s2 + state['bn2.bias']

    # 4. DW3
    w_dw3 = state['dw3.weight']
    b_dw3 = state['dw3.bias']

    # 5. PW3 + BN3
    s3 = state['bn3.weight'] / torch.sqrt(state['bn3.running_var'] + eps)
    w_pw3 = state['pw3.weight'] * s3.view(-1, 1, 1, 1)
    b_pw3 = (state['pw3.bias'] - state['bn3.running_mean']) * s3 + state['bn3.bias']

    # 6. FC
    w_fc = state['fc.weight']
    b_fc = state['fc.bias']

    # Quantize to INT8
    w_c1_q, s_c1 = quantize_per_channel(w_c1)
    w_dw2_q, s_dw2 = quantize_per_channel(w_dw2)
    w_pw2_q, s_pw2 = quantize_per_channel(w_pw2)
    w_dw3_q, s_dw3 = quantize_per_channel(w_dw3)
    w_pw3_q, s_pw3 = quantize_per_channel(w_pw3)
    w_fc_q, s_fc = quantize_per_channel(w_fc)

    total_int8_weights = len(w_c1_q) + len(w_dw2_q) + len(w_pw2_q) + len(w_dw3_q) + len(w_pw3_q) + len(w_fc_q)
    print(f"Total INT8 weights: {total_int8_weights:,} bytes ({total_int8_weights/1024:.2f} KB)")

    header_content = f"""// Tự động tạo bởi export_quantized_header.py cho ESP32-S3
// Mô hình DSCNN KWS LƯỢNG TỬ HÓA INT8 (Per-channel Symmetric INT8 Quantization)
// Giảm dung lượng trọng số từ 15.1 KB (FP32) xuống {total_int8_weights / 1024:.2f} KB (INT8) - Tiết kiệm ~75% bộ nhớ Flash
#ifndef KWS_MODEL_QUANTIZED_H
#define KWS_MODEL_QUANTIZED_H

#include <stdint.h>
#include <math.h>
#include "kws_config.h"

#define KWS_MODEL_TYPE_NAME "INT8 Quantized (Per-Channel)"

// ==========================================
// 1. TRỌNG SỐ LƯỢNG TỬ HÓA INT8 & BIAS / SCALE
// ==========================================

// Conv1: 16 filters x (1 x 3 x 3) = 144 bytes
static const int8_t w_c1_int8[144] = {{
{format_int8_array(w_c1_q)}
}};
static const float s_c1[16] = {{
{format_float_array(s_c1)}
}};
static const float b_c1_int8[16] = {{
{format_float_array(b_c1.cpu().numpy())}
}};

// Block 2: DW2 (16 ch x 3 x 3 = 144 bytes)
static const int8_t w_dw2_int8[144] = {{
{format_int8_array(w_dw2_q)}
}};
static const float s_dw2[16] = {{
{format_float_array(s_dw2)}
}};
static const float b_dw2_int8[16] = {{
{format_float_array(b_dw2.cpu().numpy())}
}};

// Block 2: PW2 (32 out x 16 in x 1 x 1 = 512 bytes)
static const int8_t w_pw2_int8[512] = {{
{format_int8_array(w_pw2_q)}
}};
static const float s_pw2[32] = {{
{format_float_array(s_pw2)}
}};
static const float b_pw2_int8[32] = {{
{format_float_array(b_pw2.cpu().numpy())}
}};

// Block 3: DW3 (32 ch x 3 x 3 = 288 bytes)
static const int8_t w_dw3_int8[288] = {{
{format_int8_array(w_dw3_q)}
}};
static const float s_dw3[32] = {{
{format_float_array(s_dw3)}
}};
static const float b_dw3_int8[32] = {{
{format_float_array(b_dw3.cpu().numpy())}
}};

// Block 3: PW3 (64 out x 32 in x 1 x 1 = 2048 bytes)
static const int8_t w_pw3_int8[2048] = {{
{format_int8_array(w_pw3_q)}
}};
static const float s_pw3[64] = {{
{format_float_array(s_pw3)}
}};
static const float b_pw3_int8[64] = {{
{format_float_array(b_pw3.cpu().numpy())}
}};

// FC: 10 classes x 64 in = 640 bytes
static const int8_t w_fc_int8[640] = {{
{format_int8_array(w_fc_q)}
}};
static const float s_fc[10] = {{
{format_float_array(s_fc)}
}};
static const float b_fc_int8[10] = {{
{format_float_array(b_fc.cpu().numpy())}
}};

// ==========================================
// 2. HÀM SUY LUẬN MÔ HÌNH INT8 (INFERENCE ENGINE)
// ==========================================
inline void kws_run_inference_quantized(const float* mfcc_in, float* out_probs) {{
    // Kích thước các lớp trung gian
    // pool1: 16 x 49 x 6 = 4704 floats
    // pool2: 32 x 24 x 3 = 2304 floats
    // pool3: 64 x 12 x 1 = 768 floats
    static float pool1[16 * 49 * 6];
    float* pool2 = pool1;
    float* pool3 = pool1;
    static float dw_buf[32 * 49 * 6];

    // 1. Conv1 (1 -> 16, 3x3 pad 1) + ReLU + MaxPool(2,2)
    for (int oc = 0; oc < 16; oc++) {{
        const int8_t* w_k = &w_c1_int8[oc * 9];
        float scale_k = s_c1[oc];
        float bias = b_c1_int8[oc];

        for (int pi = 0; pi < 49; pi++) {{
            int i0 = pi * 2;
            for (int pj = 0; pj < 6; pj++) {{
                int j0 = pj * 2;
                float max_val = -1e9f;

                for (int di = 0; di < 2; di++) {{
                    int r = i0 + di;
                    for (int dj = 0; dj < 2; dj++) {{
                        int c = j0 + dj;
                        float dot = 0.0f;

                        for (int ki = 0; ki < 3; ki++) {{
                            int in_r = r + ki - 1;
                            if (in_r >= 0 && in_r < 99) {{
                                for (int kj = 0; kj < 3; kj++) {{
                                    int in_c = c + kj - 1;
                                    if (in_c >= 0 && in_c < 13) {{
                                        dot += mfcc_in[in_r * 13 + in_c] * (float)w_k[ki * 3 + kj];
                                    }}
                                }}
                            }}
                        }}

                        float sum = bias + dot * scale_k;
                        if (sum < 0.0f) sum = 0.0f; // ReLU
                        if (sum > max_val) max_val = sum;
                    }}
                }}
                pool1[(oc * 49 + pi) * 6 + pj] = max_val;
            }}
        }}
    }}

    // 2. Block 2: DW2 (16 ch, 3x3 pad 1)
    for (int c = 0; c < 16; c++) {{
        const int8_t* w_k = &w_dw2_int8[c * 9];
        float scale_k = s_dw2[c];
        float bias = b_dw2_int8[c];

        for (int i = 0; i < 49; i++) {{
            for (int j = 0; j < 6; j++) {{
                float dot = 0.0f;
                for (int ki = 0; ki < 3; ki++) {{
                    int in_r = i + ki - 1;
                    if (in_r >= 0 && in_r < 49) {{
                        for (int kj = 0; kj < 3; kj++) {{
                            int in_c = j + kj - 1;
                            if (in_c >= 0 && in_c < 6) {{
                                dot += pool1[(c * 49 + in_r) * 6 + in_c] * (float)w_k[ki * 3 + kj];
                            }}
                        }}
                    }}
                }}
                dw_buf[(c * 49 + i) * 6 + j] = bias + dot * scale_k;
            }}
        }}
    }}

    // PW2 (16 -> 32, 1x1) + ReLU + MaxPool(2,2)
    for (int oc = 0; oc < 32; oc++) {{
        float bias = b_pw2_int8[oc];
        float scale_k = s_pw2[oc];
        const int8_t* pw_weights = &w_pw2_int8[oc * 16];

        for (int pi = 0; pi < 24; pi++) {{
            int i0 = pi * 2;
            for (int pj = 0; pj < 3; pj++) {{
                int j0 = pj * 2;
                float max_val = -1e9f;

                for (int di = 0; di < 2; di++) {{
                    int r = i0 + di;
                    for (int dj = 0; dj < 2; dj++) {{
                        int c = j0 + dj;
                        float dot = 0.0f;
                        for (int ic = 0; ic < 16; ic++) {{
                            dot += dw_buf[(ic * 49 + r) * 6 + c] * (float)pw_weights[ic];
                        }}
                        float sum = bias + dot * scale_k;
                        if (sum < 0.0f) sum = 0.0f; // ReLU
                        if (sum > max_val) max_val = sum;
                    }}
                }}
                pool2[(oc * 24 + pi) * 3 + pj] = max_val;
            }}
        }}
    }}

    // 3. Block 3: DW3 (32 ch, 3x3 pad 1)
    for (int c = 0; c < 32; c++) {{
        const int8_t* w_k = &w_dw3_int8[c * 9];
        float scale_k = s_dw3[c];
        float bias = b_dw3_int8[c];

        for (int i = 0; i < 24; i++) {{
            for (int j = 0; j < 3; j++) {{
                float dot = 0.0f;
                for (int ki = 0; ki < 3; ki++) {{
                    int in_r = i + ki - 1;
                    if (in_r >= 0 && in_r < 24) {{
                        for (int kj = 0; kj < 3; kj++) {{
                            int in_c = j + kj - 1;
                            if (in_c >= 0 && in_c < 3) {{
                                dot += pool2[(c * 24 + in_r) * 3 + in_c] * (float)w_k[ki * 3 + kj];
                            }}
                        }}
                    }}
                }}
                dw_buf[(c * 24 + i) * 3 + j] = bias + dot * scale_k;
            }}
        }}
    }}

    // PW3 (32 -> 64, 1x1) + ReLU + MaxPool(2,2)
    for (int oc = 0; oc < 64; oc++) {{
        float bias = b_pw3_int8[oc];
        float scale_k = s_pw3[oc];
        const int8_t* pw_weights = &w_pw3_int8[oc * 32];

        for (int pi = 0; pi < 12; pi++) {{
            int i0 = pi * 2;
            for (int pj = 0; pj < 1; pj++) {{
                int j0 = pj * 2;
                float max_val = -1e9f;

                for (int di = 0; di < 2; di++) {{
                    int r = i0 + di;
                    for (int dj = 0; dj < 2; dj++) {{
                        int c = j0 + dj;
                        if (c >= 3) continue;
                        float dot = 0.0f;
                        for (int ic = 0; ic < 32; ic++) {{
                            dot += dw_buf[(ic * 24 + r) * 3 + c] * (float)pw_weights[ic];
                        }}
                        float sum = bias + dot * scale_k;
                        if (sum < 0.0f) sum = 0.0f; // ReLU
                        if (sum > max_val) max_val = sum;
                    }}
                }}
                pool3[oc * 12 + pi] = max_val;
            }}
        }}
    }}

    // 4. Global Average Pooling (64 ch x 12 -> 64)
    float gap[64];
    for (int c = 0; c < 64; c++) {{
        float sum = 0.0f;
        for (int i = 0; i < 12; i++) {{
            sum += pool3[c * 12 + i];
        }}
        gap[c] = sum / 12.0f;
    }}

    // 5. Fully Connected (64 -> NUM_CLASSES) + Softmax
    float logits[NUM_CLASSES];
    float max_logit = -1e9f;
    for (int k = 0; k < NUM_CLASSES; k++) {{
        float bias = b_fc_int8[k];
        float scale_k = s_fc[k];
        const int8_t* fc_row = &w_fc_int8[k * 64];

        float dot = 0.0f;
        for (int c = 0; c < 64; c++) {{
            dot += gap[c] * (float)fc_row[c];
        }}
        float sum = bias + dot * scale_k;
        logits[k] = sum;
        if (sum > max_logit) max_logit = sum;
    }}

    float sum_exp = 0.0f;
    for (int k = 0; k < NUM_CLASSES; k++) {{
        out_probs[k] = expf(logits[k] - max_logit);
        sum_exp += out_probs[k];
    }}
    for (int k = 0; k < NUM_CLASSES; k++) {{
        out_probs[k] /= sum_exp;
    }}
}}

#endif // KWS_MODEL_QUANTIZED_H
"""

    with open(OUTPUT_HEADER, "w", encoding="utf-8") as f:
        f.write(header_content)

    print(f"-> Generated {OUTPUT_HEADER} successfully!")

if __name__ == "__main__":
    export_quantized_header()
