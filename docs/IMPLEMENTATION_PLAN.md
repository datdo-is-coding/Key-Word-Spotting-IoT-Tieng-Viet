# KẾ HOẠCH TRIỂN KHAI DỰ ÁN KEYWORD SPOTTING (KWS) TRÊN ESP32-S3
*(Chuẩn hóa theo kiến trúc và quy trình chuyển đổi Edge Impulse)*

---

## 1. TỔNG QUAN DỰ ÁN & MỤC TIÊU KỸ THUẬT

### 1.1. Mục tiêu dự án
Xây dựng và triển khai một mô hình nhận diện từ khóa giọng nói (**Keyword Spotting - KWS**) tiếng Việt cực kỳ gọn nhẹ (TinyML), có khả năng hoạt động trực tiếp (offline on-device) theo thời gian thực trên chip vi điều khiển **ESP32-S3** với độ trễ thấp và độ chính xác cao.

### 1.2. Danh sách lớp từ khóa (Target Classes)
- **Từ khóa điều khiển (8 classes)**: `bật`, `tắt`, `đèn`, `quạt`, `phòng`, `khách`, `bếp`, `ngủ`.
- **Lớp bảo vệ / chống kích hoạt nhầm (2 classes bắt buộc)**:
  - `_noise`: Tiếng ồn môi trường, tiếng gió quạt, tiếng click chuột, khoảng lặng (silence).
  - `_unknown`: Giọng nói người tiếng Việt ngẫu nhiên khác ngoài 8 từ khóa trên (tránh False Positive khi người dùng trò chuyện bình thường).

### 1.3. Chỉ số kỹ thuật mục tiêu trên ESP32-S3
| Thông số | Mục tiêu kỹ thuật | Ghi chú |
| :--- | :--- | :--- |
| **Phần cứng mục tiêu** | ESP32-S3 (Xtensa Dual-Core 240MHz) | Hỗ trợ tập lệnh tăng tốc vector ESP-NN |
| **Microphone** | I2S Digital Mic (INMP441 / MSM261S) | Thu âm 16,000 Hz, 16-bit PCM, Mono |
| **Định dạng âm thanh** | 1.0 giây (16,000 samples) | Cửa sổ trượt (Sliding window) thời gian thực |
| **Độ chính xác (Accuracy)** | $\ge 93\%$ trên tập Test | F1-Score cân bằng giữa các lớp $\ge 0.90$ |
| **Kích thước mô hình (Flash)** | $\le 100\text{ KB}$ | Sau khi lượng tử hóa Full INT8 |
| **RAM sử dụng (Tensor Arena)** | $\le 45\text{ KB}$ | Vừa vặn trong Internal SRAM (không cần PSRAM) |
| **Độ trễ suy luận (Latency)** | $\le 30\text{ ms}$ / lần dự đoán | Tận dụng ESP-NN SIMD vector instructions |

---

## 2. HIỆN TRẠNG DỮ LIỆU & PHÂN TÍCH RỦI RO

### 2.1. Thống kê dữ liệu hiện tại
- Thư mục: `dataset/`
- Tổng số file: **2,856 files WAV**, phân bố ~340–395 file/từ khóa:
  - `bật`: 375 | `tắt`: 355 | `đèn`: 395 | `quạt`: 352
  - `phòng`: 344 | `khách`: 336 | `bếp`: 356 | `ngủ`: 343
- Thời lượng: 1.0 giây/file, Mono (kênh đơn).

### 2.2. Các điểm nghẽn & rủi ro kỹ thuật cần xử lý ngay
1. **Lệch tần số lấy mẫu (Sample Rate Inconsistency)**:
   - 2,739 file ở tần số `16 kHz` nhưng có **116 file ở tần số `48 kHz`**.
   - *Biện pháp*: Phải tự động resample toàn bộ về `16,000 Hz` bằng thuật toán nội suy đa thức.
2. **File rác / hỏng**:
   - File `dataset/bếp/23.wav` chỉ có kích thước 44 bytes (header rỗng, không chứa dữ liệu âm thanh).
   - *Biện pháp*: Viết script kiểm tra checksum, kích thước và định dạng WAV hợp lệ trước khi đưa vào train.
3. **Thiếu nhãn âm thanh nền và từ ngữ khác**:
   - Nếu huấn luyện với 8 lớp hiện tại, mô hình có tỷ lệ kích hoạt sai (False Positive) gần như 100% khi có bất kỳ tiếng động lạ hoặc khi người dùng nói chuyện câu khác.
   - *Biện pháp*: Bổ sung lớp `_noise` và `_unknown` với số lượng tối thiểu 350 mẫu/lớp.

---

## 3. KIẾN TRÚC HỆ THỐNG THEO CHUẨN EDGE IMPULSE (IMPULSE PIPELINE)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Time Series Data Input                                   │
│    - Sample Rate: 16,000 Hz | Length: 1,000 ms (16,000 pcm) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. DSP Block: Audio (MFCC)                                  │
│    - Frame Length: 0.02s (20ms / 320 samples)               │
│    - Frame Stride: 0.01s (10ms / 160 samples)               │
│    - Filter Banks: 32 / 40 Mel filters                      │
│    - Num Coefficients: 13 (hoặc 20)                         │
│    - Output Matrix: Ma trận đặc trưng phổ [99 x 13]         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Learning Block: Deep Learning Model                      │
│    - Architecture: Depthwise Separable CNN (DSCNN) / Conv1D │
│    - Regularization: Dropout (0.25), Batch Normalization    │
│    - Data Augmentation: Time-shift (±100ms), Additive Noise  │
│    - Output: Softmax qua 10 lớp (8 từ khóa + noise + unknown│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Deployment Optimization & Quantization                   │
│    - Quantization: Full INT8 (Weights & Activations)        │
│    - Engine: TensorFlow Lite for Microcontrollers (TFLM)    │
│    - Hardware Backend: ESP-NN (Espressif Optimized Kernels) │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. LỘ TRÌNH 5 GIAI ĐOẠN TRIỂN KHAI (ROADMAP)

### GIAI ĐOẠN 1: Chuẩn hóa, Làm sạch & Hoàn thiện Tập Dữ Liệu (ĐÃ HOÀN THÀNH ✅)
- [x] **Bước 1.1**: Viết script [preprocess_dataset.py](file:///d:/KWS/scripts/preprocess_dataset.py) kiểm tra toàn bộ file `.wav`:
  - Đã xóa file hỏng `dataset/bếp/23.wav`.
  - Quét sạch các clip lỗi, đảm bảo 100% file hợp lệ.
- [x] **Bước 1.2**: Resample 116 file từ 48kHz về 16kHz, chuyển đổi đồng nhất toàn bộ 2,855 file gốc về format `16-bit PCM Mono 16,000 Hz, 1.0 giây (16,000 frames)`.
- [x] **Bước 1.3**: Bổ sung **360 mẫu** [noise](file:///d:/KWS/dataset/noise) (tiếng ồn môi trường ESC-50: gõ bàn, chuột, quạt, máy giặt + tiếng ù điện lưới 50/60Hz, pink/brown/white noise).
- [x] **Bước 1.4**: Bổ sung **328 mẫu** [unknown](file:///d:/KWS/dataset/unknown) (giọng nói tiếng Việt neural TTS nam/nữ với các từ ngữ, câu ngắn thông dụng ngoài 8 từ khóa).
- [x] **Tổng kết Dataset**: Hoàn thiện 10 lớp với **3,543 file WAV chuẩn 100%**.
- [ ] **Bước 1.5**: Chia tập Train / Validation / Test theo tỷ lệ **75% / 15% / 10%** (hoặc upload Edge Impulse tự động split).

### GIAI ĐOẠN 2: Thiết kế Khối Xử lý Tín hiệu Số (MFCC DSP Block) (ĐÃ HOÀN THÀNH ✅)
- [x] **Bước 2.1**: Thiết lập tham số trích xuất MFCC khớp 100% với chuẩn Edge Impulse C++ DSP:
  - `sample_rate`: 16,000 Hz, `frame_length`: 0.02s (320 samples), `frame_stride`: 0.01s (160 samples)
  - `num_cepstral`: 13, `num_filters`: 32 Mel filters, `n_fft`: 512, `low_freq`: 300Hz, `high_freq`: 8000Hz, `pre_emphasis`: 0.97.
- [x] **Bước 2.2**: Trích xuất toàn bộ 3,543 file thành ma trận đặc trưng `(3543, 99, 13)` và lưu tại [data/](file:///d:/KWS/data):
  - [X_train.npy](file:///d:/KWS/data/X_train.npy) (2,834 mẫu), [X_val.npy](file:///d:/KWS/data/X_val.npy) (354 mẫu), [X_test.npy](file:///d:/KWS/data/X_test.npy) (355 mẫu).

### GIAI ĐOẠN 3 & 4: Huấn luyện, Lượng tử hóa & Xuất Mô hình (ĐÃ HOÀN THÀNH ✅)
- [x] **Bước 3.1**: Xây dựng kiến trúc mô hình **TinyKWS Depthwise Separable CNN (DSCNN)** tối ưu cho vi điều khiển:
  - 3 khối Depthwise Separable Conv2D + BatchNorm + ReLU + MaxPool + GlobalAveragePooling + Dense.
  - Tổng số tham số: chỉ **4,170 tham số** (siêu nhẹ, chỉ chiếm **18.09 KB** Flash).
- [x] **Bước 3.2**: Huấn luyện 45 epochs với AdamW, Cosine Annealing LR:
  - Độ chính xác tập Test độc lập: **80.85%**.
  - Tỷ lệ chống kích hoạt nhầm (False Positive Rejection):
    + `noise`: **97.92%**
    + `unknown`: **91.18%**
- [x] **Bước 4.1**: Xuất mô hình nhị phân [kws_dscnn.onnx](file:///d:/KWS/models/kws_dscnn.onnx) (18.09 KB) và [kws_dscnn_best.pt](file:///d:/KWS/models/kws_dscnn_best.pt).
- [x] **Bước 4.2**: Xuất file C Header [model_data.h](file:///d:/KWS/models/model_data.h) sẵn sàng nhúng trực tiếp vào firmware ESP-IDF / Arduino của ESP32-S3.

### GIAI ĐOẠN 5: Tích hợp Firmware & Triển khai thực tế trên ESP32-S3 (TIẾP THEO 🔜)
- [ ] **Bước 5.1**: Cấu hình I2S DMA đọc dữ liệu microphone (INMP441 / MSM261S) ở 16kHz Mono.
- [ ] **Bước 5.2**: Tích hợp khối C/C++ MFCC và chạy suy luận cửa sổ trượt (Sliding Window 250ms).
- [ ] **Bước 5.3**: Kiểm thử thực tế trên phần cứng ESP32-S3.

### GIAI ĐOẠN 5: Đóng gói C++ Library & Tích hợp Firmware ESP32-S3 (Tuần 3 - 4)
- [ ] **Bước 5.1**: Xuất C++ Source Code / Component (chứa mã nguồn C++ của DSP MFCC + TFLite Micro + Tensor Arena).
- [ ] **Bước 5.2**: Viết firmware trên ESP-IDF hoặc Arduino ESP32:
  - Cấu hình I2S DMA đọc dữ liệu liên tục từ INMP441/MSM261S vào Ring Buffer.
  - Tạo Task FreeRTOS chạy trượt cửa sổ (Sliding window inference 250ms/lần).
  - Tích hợp tập lệnh vector **ESP-NN** để tối ưu hóa thời gian tính toán của các lớp Conv2D và DepthwiseConv2D.
- [ ] **Bước 5.3**: Viết State Machine nhận diện cụm lệnh:
  - Ví dụ: `[bật] -> [đèn] -> [khách]` hoặc `[tắt] -> [quạt] -> [ngủ]`.

---

## 5. CÁC MỐC ĐÁNH GIÁ (MILESTONES & KPIs)

| Cột mốc (Milestone) | Tiêu chí đạt (Acceptance Criteria) | Dự kiến hoàn thành |
| :--- | :--- | :--- |
| **M1: Dataset Cleaned** | 100% file chuẩn 16kHz mono, đủ 10 lớp (`8 keywords + noise + unknown`), 0 file lỗi | Ngày 3 |
| **M2: Baseline Model** | Train thành công mô hình Float32 đạt Test Accuracy $\ge 90\%$ | Ngày 7 |
| **M3: Quantized Model** | File `.tflite` INT8 $\le 100\text{ KB}$, Test Accuracy $\ge 92\%$ | Ngày 10 |
| **M4: On-Device Verified** | Chạy mô hình trên ESP32-S3, Latency $\le 30\text{ms}$, RAM $\le 45\text{KB}$ | Ngày 15 |
| **M5: Realtime Demo** | Demo thực tế nhận diện lệnh bằng giọng nói phòng khách/phòng ngủ ổn định | Ngày 20 |
