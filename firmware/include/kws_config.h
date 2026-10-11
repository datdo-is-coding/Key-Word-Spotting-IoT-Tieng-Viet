#ifndef KWS_CONFIG_H
#define KWS_CONFIG_H

#include <driver/i2s.h>
#include <stdint.h>

// ==========================================
// 1. CẤU HÌNH PHẦN CỨNG MICROPHONE I2S (INMP441)
// ==========================================
#define I2S_PORT I2S_NUM_0
#define MIC_I2S_GPIO_BCLK GPIO_NUM_15 // Bit Clock (BCLK / SCK) -> IO15
#define MIC_I2S_GPIO_WS GPIO_NUM_16   // Word Select (WS / LRCK) -> IO16
#define MIC_I2S_GPIO_DIN GPIO_NUM_8 // Serial Data Out (SD / DOUT / DIN) -> IO8

// ==========================================
// 2. CẤU HÌNH RELAY ĐIỀU KHIỂN THIẾT BỊ
// ==========================================
#define RELAY_1_PIN GPIO_NUM_4 // Relay 1 -> IO4: Điều khiển BẬT / TẮT ĐÈN
#define RELAY_2_PIN GPIO_NUM_5 // Relay 2 -> IO5: Điều khiển BẬT / TẮT QUẠT

// Mức logic tích cực đóng Relay: Active-LOW (0 là ON, 1 là OFF)
#define RELAY_ON_LEVEL LOW
#define RELAY_OFF_LEVEL HIGH

// Chân LED báo trạng thái nhận diện (tùy chọn)
#define LED_STATUS_PIN 2

// ==========================================
// 3. CẤU HÌNH THÔNG SỐ ÂM THANH & CỬA SỔ TRƯỢT
// ==========================================
#define SAMPLE_RATE 16000    // Tần số lấy mẫu 16,000 Hz
#define AUDIO_WINDOW_MS 1000 // Cửa sổ nhận diện 1.0 giây
#define AUDIO_STEP_MS 250    // Bước trượt 250ms (chạy suy luận 4 lần/giây)

#define TOTAL_SAMPLES (SAMPLE_RATE * AUDIO_WINDOW_MS / 1000) // 16,000 mẫu
#define STEP_SAMPLES (SAMPLE_RATE * AUDIO_STEP_MS / 1000)    // 4,000 mẫu

// ==========================================
// 4. CẤU HÌNH KHỐI DSP MFCC
// ==========================================
#define FRAME_LEN_SAMPLES 320  // 20ms @ 16kHz
#define FRAME_STEP_SAMPLES 160 // 10ms @ 16kHz
#define FFT_SIZE 512           // Kích thước FFT
#define NUM_MEL_FILTERS 32     // 32 dải lọc Mel
#define NUM_MFCC_COEFFS 13     // 13 hệ số MFCC
#define NUM_FRAMES 99          // 99 khung thời gian trong 1.0 giây

// Tham số chuẩn hóa Z-Score từ tập huấn luyện
#define NORM_MEAN (-9.295776f)
#define NORM_STD (35.392517f)

// ==========================================
// 5. DANH SÁCH 10 NHÃN NHẬN DIỆN
// ==========================================
#define NUM_CLASSES 10

enum KwsClassId {
    CLASS_BAT     = 0, // "bật"
    CLASS_BEP     = 1, // "bếp"
    CLASS_KHACH   = 2, // "khách"
    CLASS_NGU     = 3, // "ngủ"
    CLASS_NOISE   = 4, // "noise"
    CLASS_PHONG   = 5, // "phòng"
    CLASS_QUAT    = 6, // "quạt"
    CLASS_TAT     = 7, // "tắt"
    CLASS_UNKNOWN = 8, // "unknown"
    CLASS_DEN     = 9  // "đèn"
};

static const char *const KWS_LABELS[NUM_CLASSES] = {
    "bật",     // 0
    "bếp",     // 1
    "khách",   // 2
    "ngủ",     // 3
    "noise",   // 4
    "phòng",   // 5
    "quạt",    // 6
    "tắt",     // 7
    "unknown", // 8
    "đèn"      // 9
};

// ==========================================
// 6. NGƯỠNG KÍCH HOẠT & LỌC NHIỄU (THRESHOLDS)
// ==========================================
#define CONFIDENCE_THRESHOLD 0.40f // Ngưỡng xác suất tối thiểu kích hoạt lệnh (40%)
#define COOLDOWN_MS 500          // Thời gian chờ tránh kích hoạt lặp từ

#endif // KWS_CONFIG_H
