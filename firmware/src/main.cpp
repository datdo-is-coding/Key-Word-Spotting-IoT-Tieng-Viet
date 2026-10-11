#include "kws_config.h"
#include "mfcc.h"
#include "model_data.h"
#include "kws_model.h"
#include "audio_input.h"
#include <Arduino.h>

// ==========================================
// ĐỐI TƯỢNG TOÀN CỤC & BỘ ĐỆM ÂM THANH
// ==========================================
static MFCCExtractor mfcc_extractor;

// Bộ đệm âm thanh 1.0 giây (16,000 mẫu float)
static float audio_buffer[TOTAL_SAMPLES];
// Bộ đệm đặc trưng MFCC (99 frames x 13 coeffs = 1,287 giá trị)
static float mfcc_features[NUM_FRAMES * NUM_MFCC_COEFFS];

// Thời gian cooldown giữa các lần nhận diện
static unsigned long last_detection_time = 0;

// Trạng thái Relay hiện tại
static bool lamp_state = false; // Relay 1 (IO4)
static bool fan_state = false;  // Relay 2 (IO5)

// Trạng thái chuỗi lệnh giọng nói
static int pending_action = -1;  // CLASS_BAT (0) hoặc CLASS_TAT (7)
static int pending_device = -1;  // CLASS_DEN (9) hoặc CLASS_QUAT (6)
static unsigned long state_timeout = 0;

// ==========================================
// HÀM ĐIỀU KHIỂN RELAY THỰC TẾ
// ==========================================
void set_lamp_relay(bool turn_on) {
  lamp_state = turn_on;
  digitalWrite(RELAY_1_PIN, lamp_state ? RELAY_ON_LEVEL : RELAY_OFF_LEVEL);

  Serial.println("\n***************************************************");
  Serial.printf("💡 [RELAY 1 - ĐÈN (IO4)] => %s (Chân IO4 = %d)\n",
                lamp_state ? "ĐÃ BẬT (ON)" : "ĐÃ TẮT (OFF)",
                lamp_state ? RELAY_ON_LEVEL : RELAY_OFF_LEVEL);
  Serial.println("***************************************************\n");

  // Nháy LED trạng thái xác nhận
  digitalWrite(LED_STATUS_PIN, HIGH);
  delay(150);
  digitalWrite(LED_STATUS_PIN, LOW);
}

void set_fan_relay(bool turn_on) {
  fan_state = turn_on;
  digitalWrite(RELAY_2_PIN, fan_state ? RELAY_ON_LEVEL : RELAY_OFF_LEVEL);

  Serial.println("\n***************************************************");
  Serial.printf("🌀 [RELAY 2 - QUẠT (IO5)] => %s (Chân IO5 = %d)\n",
                fan_state ? "ĐÃ BẬT (ON)" : "ĐÃ TẮT (OFF)",
                fan_state ? RELAY_ON_LEVEL : RELAY_OFF_LEVEL);
  Serial.println("***************************************************\n");

  // Nháy LED trạng thái xác nhận
  digitalWrite(LED_STATUS_PIN, HIGH);
  delay(150);
  digitalWrite(LED_STATUS_PIN, LOW);
}

// ==========================================
// CẤU HÌNH VAD TÁCH ÂM TIẾT TỰ ĐỘNG (METHOD 2)
// ==========================================
#define VAD_SLICE_MS         50     // 50ms mỗi lát đọc âm thanh
#define VAD_SLICE_SAMPLES    (SAMPLE_RATE * VAD_SLICE_MS / 1000) // 800 mẫu @ 16kHz
#define VAD_PREROLL_SLICES   2      // 100ms đệm trước (tránh mất phụ âm đầu b, t, đ)
#define VAD_MIN_SPEECH_SAMP  (SAMPLE_RATE * 150 / 1000) // 2,400 mẫu (150ms tối thiểu)
#define VAD_MAX_SPEECH_SAMP  (SAMPLE_RATE * 800 / 1000) // 12,800 mẫu (800ms tối đa)
#define VAD_SILENCE_SLICES   2      // 100ms (2 x 50ms) im lặng liên tiếp để chốt kết thúc từ

enum VADState {
  VAD_IDLE,       // Đang chờ người dùng nói
  VAD_SPEAKING    // Đang gom âm tiết của từ
};

static VADState vad_state = VAD_IDLE;
static float preroll_buf[VAD_PREROLL_SLICES * VAD_SLICE_SAMPLES];
static float slice_buf[VAD_SLICE_SAMPLES];
static int recorded_samples = 0;
static int silence_slice_count = 0;

// ==========================================
// XỬ LÝ LỆNH ĐIỀU KHIỂN & CHUYỂN MẠCH RELAY
// ==========================================
void handle_detected_keyword(int class_id, float confidence) {
  if (class_id < 0 || class_id >= NUM_CLASSES) return;

  // Bỏ qua tiếng ồn hoặc câu nói ngẫu nhiên
  if (class_id == CLASS_NOISE || class_id == CLASS_UNKNOWN) {
    return;
  }

  const char *label = KWS_LABELS[class_id];
  Serial.printf("🎯 [NHẬN DIỆN] Từ khóa: '%s' (Độ tin cậy: %.1f%%)\n",
                label, confidence * 100.0f);

  // Kiểm tra timeout chuỗi lệnh (quá 5 giây không có từ tiếp theo thì reset)
  if ((pending_action != -1 || pending_device != -1) && millis() > state_timeout) {
    Serial.println("  ⚠️ [TIMEOUT] Quá 5 giây chờ lệnh tiếp theo, hủy chuỗi lệnh!");
    pending_action = -1;
    pending_device = -1;
  }

  // TRƯỜNG HỢP 1: Từ nhận diện là HÀNH ĐỘNG (bật / tắt)
  if (class_id == CLASS_BAT || class_id == CLASS_TAT) {
    pending_action = class_id;

    // Nếu trước đó đã nhận tên thiết bị ("đèn" hoặc "quạt") => THỰC THI RELAY NGAY LẬP TỨC!
    if (pending_device == CLASS_DEN) {
      bool turn_on = (pending_action == CLASS_BAT);
      set_lamp_relay(turn_on);
      pending_action = -1;
      pending_device = -1;
      return;
    } else if (pending_device == CLASS_QUAT) {
      bool turn_on = (pending_action == CLASS_BAT);
      set_fan_relay(turn_on);
      pending_action = -1;
      pending_device = -1;
      return;
    }

    // Nếu chưa có thiết bị: chờ nói thiết bị tiếp theo
    state_timeout = millis() + 5000;
    Serial.printf("  👉 [BƯỚC 1: HÀNH ĐỘNG] '%s' => Đang chờ nói tên thiết bị ('đèn' hoặc 'quạt')...\n",
                  label);
  }
  // TRƯỜNG HỢP 2: Từ nhận diện là THIẾT BỊ (đèn / quạt)
  else if (class_id == CLASS_DEN || class_id == CLASS_QUAT) {
    pending_device = class_id;

    // Nếu trước đó đã nhận hành động ("bật" hoặc "tắt") => THỰC THI RELAY NGAY LẬP TỨC!
    if (pending_action == CLASS_BAT || pending_action == CLASS_TAT) {
      bool turn_on = (pending_action == CLASS_BAT);
      if (pending_device == CLASS_DEN) {
        set_lamp_relay(turn_on);
      } else {
        set_fan_relay(turn_on);
      }
      pending_action = -1;
      pending_device = -1;
      return;
    }

    // Nếu người dùng nói thiết bị trước ("đèn" ... rồi mới "bật")
    state_timeout = millis() + 5000;
    Serial.printf("  👉 [BƯỚC 1: THIẾT BỊ] '%s' => Đang chờ nói hành động ('bật' hoặc 'tắt')...\n",
                  label);
  }
  // TRƯỜNG HỢP 3: Khu vực phòng (tùy chọn)
  else if (class_id == CLASS_KHACH || class_id == CLASS_BEP || class_id == CLASS_NGU) {
    Serial.printf("  🏠 [KHU VỰC] Phòng %s.\n", label);
  }
}

// ==========================================
// HÀM SETUP
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=======================================================");
  Serial.println("  ESP32-S3 KEYWORD SPOTTING (KWS) - VOICE RELAY CONTROL");
  Serial.println("=======================================================");
  Serial.printf("Mô hình lượng tử hóa INT8: %u bytes (%0.2f KB)\n",
                kws_model_len, (float)kws_model_len / 1024.0f);

  // Cấu hình các chân Relay đầu ra
  pinMode(RELAY_1_PIN, OUTPUT);
  pinMode(RELAY_2_PIN, OUTPUT);
  pinMode(LED_STATUS_PIN, OUTPUT);

  // Mặc định ban đầu: TẮT toàn bộ Relay
  digitalWrite(RELAY_1_PIN, RELAY_OFF_LEVEL);
  digitalWrite(RELAY_2_PIN, RELAY_OFF_LEVEL);
  digitalWrite(LED_STATUS_PIN, LOW);

  Serial.printf("[CẤU HÌNH RELAY] Relay 1 (ĐÈN): IO%d | Relay 2 (QUẠT): IO%d (0=ON, 1=OFF)\n",
                RELAY_1_PIN, RELAY_2_PIN);

  // Khởi tạo bộ đệm âm thanh và đệm Pre-roll
  memset(audio_buffer, 0, sizeof(audio_buffer));
  memset(preroll_buf, 0, sizeof(preroll_buf));

  // Khởi tạo bộ trích xuất MFCC
  if (!mfcc_extractor.init()) {
    Serial.println("[ERROR] Lỗi khởi tạo MFCC extractor!");
    while (1)
      delay(100);
  }
  Serial.println("[OK] Khối trích xuất MFCC sẵn sàng.");

  // Khởi tạo I2S Microphone & Bộ lọc sạch Biquad
  if (audio_input_init() != ESP_OK) {
    Serial.println("[ERROR] Lỗi khởi tạo micro, dừng hệ thống!");
    while (1)
      delay(100);
  }

  Serial.println("\n[SẴN SÀNG LẮNG NGHE LỆNH GIỌNG NÓI - VAD TÁCH ÂM TIẾT]");
  Serial.println("👉 Nói liền mạch: 'BẬT' ... 'ĐÈN'  => Đóng Relay 1 (IO4)");
  Serial.println("👉 Nói liền mạch: 'TẮT' ... 'ĐÈN'  => Ngắt Relay 1 (IO4)");
  Serial.println("👉 Nói liền mạch: 'BẬT' ... 'QUẠT' => Đóng Relay 2 (IO5)");
  Serial.println("👉 Nói liền mạch: 'TẮT' ... 'QUẠT' => Ngắt Relay 2 (IO5)\n");
}

// ==========================================
// HÀM LOOP CHÍNH: VAD SYLLABLE ENGINE (PHƯƠNG PHÁP 2)
// ==========================================
void loop() {
  // 1. Đọc một lát âm thanh 50ms (800 mẫu) qua Biquad HPF+LPF
  float slice_rms = 0.0f;
  audio_input_read_clean(slice_buf, VAD_SLICE_SAMPLES, &slice_rms);

  float ambient = audio_input_get_ambient_rms();
  float speech_thresh = ambient * 2.2f + 0.006f;
  float silence_thresh = ambient * 1.3f + 0.002f;

  static unsigned long last_listen_report = 0;

  // 2. VAD State Machine: Tự động tách và căn giữa âm tiết
  if (vad_state == VAD_IDLE) {
    // Cuộn vòng bộ đệm Pre-roll 100ms (2 lát x 50ms)
    memmove(&preroll_buf[0], &preroll_buf[VAD_SLICE_SAMPLES], VAD_SLICE_SAMPLES * sizeof(float));
    memcpy(&preroll_buf[VAD_SLICE_SAMPLES], slice_buf, VAD_SLICE_SAMPLES * sizeof(float));

    // Định kỳ gửi thông báo "Đang nghe" mỗi 2 giây khi phòng yên tĩnh
    if (millis() - last_listen_report >= 2000) {
      last_listen_report = millis();
      Serial.printf("[ESP32-S3] 🎙️ Đang lắng nghe... (RMS: %.4f | Ồn nền: %.4f)\n", slice_rms, ambient);
    }

    // Phát hiện bắt đầu nói âm tiết
    if (slice_rms >= speech_thresh) {
      vad_state = VAD_SPEAKING;
      recorded_samples = 0;
      silence_slice_count = 0;

      // Nạp 100ms Pre-roll vào đầu để bảo toàn phụ âm đầu (b, t, đ)
      memcpy(&audio_buffer[0], preroll_buf, sizeof(preroll_buf));
      recorded_samples += VAD_PREROLL_SLICES * VAD_SLICE_SAMPLES;

      // Nạp lát âm thanh hiện tại
      memcpy(&audio_buffer[recorded_samples], slice_buf, VAD_SLICE_SAMPLES * sizeof(float));
      recorded_samples += VAD_SLICE_SAMPLES;

      Serial.printf("\n⚡ [BẮT ĐẦU TỪ] RMS: %.4f (Ngưỡng nói: %.4f)\n", slice_rms, speech_thresh);
    }
  } else if (vad_state == VAD_SPEAKING) {
    // Tiếp tục gom âm thanh vào audio_buffer nếu còn chỗ
    if (recorded_samples + VAD_SLICE_SAMPLES <= VAD_MAX_SPEECH_SAMP) {
      memcpy(&audio_buffer[recorded_samples], slice_buf, VAD_SLICE_SAMPLES * sizeof(float));
      recorded_samples += VAD_SLICE_SAMPLES;
    }

    // Đếm khoảng lặng giữa 2 từ (ngắt âm tự nhiên ~80-120ms)
    if (slice_rms < silence_thresh) {
      silence_slice_count++;
    } else {
      silence_slice_count = 0;
    }

    // Kết thúc âm tiết khi: Có 100ms im lặng liên tiếp HOẶC chạm giới hạn 800ms
    bool is_word_ended = (silence_slice_count >= VAD_SILENCE_SLICES) ||
                         (recorded_samples >= VAD_MAX_SPEECH_SAMP);

    if (is_word_ended) {
      int duration_ms = (recorded_samples * 1000) / SAMPLE_RATE;
      Serial.printf("🛑 [KẾT THÚC TỪ] Thời lượng: %d ms (%d mẫu)\n", duration_ms, recorded_samples);

      // Kiểm tra độ dài hợp lệ (>= 150ms để loại trừ tiếng click / tiếng tặc lưỡi)
      if (recorded_samples >= VAD_MIN_SPEECH_SAMP) {
        // CĂN GIỮA ÂM TIẾT VÀO BUFFER 1.0S (16,000 MẪU) CHUẨN XÁC NHƯ LÚC TRAIN!
        int pad_total = TOTAL_SAMPLES - recorded_samples;
        int pad_front = pad_total / 2;

        memmove(&audio_buffer[pad_front], &audio_buffer[0], recorded_samples * sizeof(float));
        memset(&audio_buffer[0], 0, pad_front * sizeof(float));
        memset(&audio_buffer[pad_front + recorded_samples], 0, (TOTAL_SAMPLES - pad_front - recorded_samples) * sizeof(float));

        // Trích xuất MFCC từ âm tiết đã căn giữa
        mfcc_extractor.extract(audio_buffer, mfcc_features);

        // Chạy suy luận đúng 1 lần duy nhất (~88ms)
        unsigned long t_start = millis();
        float probs[NUM_CLASSES];
        kws_run_inference(mfcc_features, probs);
        unsigned long infer_duration = millis() - t_start;

        // Tìm từ khóa xác suất cao nhất
        int top_class = 0;
        float top_prob = probs[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
          if (probs[i] > top_prob) {
            top_prob = probs[i];
            top_class = i;
          }
        }

        // Tìm từ khóa xếp thứ 2
        int second_class = (top_class == 0) ? 1 : 0;
        for (int i = 0; i < NUM_CLASSES; i++) {
          if (i != top_class && probs[i] > probs[second_class]) {
            second_class = i;
          }
        }

        // Trả kết quả qua Serial Monitor
        Serial.println("---------------------------------------------------");
        Serial.printf("🎯 [KẾT QUẢ TÁCH ÂM TIẾT] (%d ms âm | Suy luận: %lu ms)\n", duration_ms, infer_duration);
        Serial.printf("   👉 Dự đoán: '%s' | Độ tin cậy: %.1f%%\n", KWS_LABELS[top_class], top_prob * 100.0f);
        Serial.printf("   👉 Xếp thứ 2: '%s' | Độ tin cậy: %.1f%%\n", KWS_LABELS[second_class], probs[second_class] * 100.0f);
        Serial.println("---------------------------------------------------");

        // Xử lý từ khóa
        if (top_prob >= CONFIDENCE_THRESHOLD) {
          handle_detected_keyword(top_class, top_prob);
        } else {
          Serial.printf("   ⚠️ [BỎ QUA] Độ tin cậy (%.1f%%) < Ngưỡng kích hoạt (%.0f%%)\n\n",
                        top_prob * 100.0f, CONFIDENCE_THRESHOLD * 100.0f);
        }
      } else {
        Serial.printf("   ⚠️ [BỎ QUA] Âm quá ngắn (< 150ms), bỏ qua tạp âm ngắn!\n\n");
      }

      // Sẵn sàng nhận từ tiếp theo ngay lập tức (KHÔNG cần chờ cooldown cứng)!
      vad_state = VAD_IDLE;
      memset(preroll_buf, 0, sizeof(preroll_buf));
      last_listen_report = millis();
    }
  }

  // Nhận thêm lệnh test thủ công từ Serial Monitor để kiểm tra đóng/ngắt Relay
  if (Serial.available()) {
    String test_cmd = Serial.readStringUntil('\n');
    test_cmd.trim();
    if (test_cmd == "bat den" || test_cmd == "1") {
      set_lamp_relay(true);
    } else if (test_cmd == "tat den" || test_cmd == "2") {
      set_lamp_relay(false);
    } else if (test_cmd == "bat quat" || test_cmd == "3") {
      set_fan_relay(true);
    } else if (test_cmd == "tat quat" || test_cmd == "4") {
      set_fan_relay(false);
    } else if (test_cmd == "bat") {
      handle_detected_keyword(CLASS_BAT, 0.95f);
    } else if (test_cmd == "tat") {
      handle_detected_keyword(CLASS_TAT, 0.95f);
    } else if (test_cmd == "den") {
      handle_detected_keyword(CLASS_DEN, 0.95f);
    } else if (test_cmd == "quat") {
      handle_detected_keyword(CLASS_QUAT, 0.95f);
    }
  }
}
