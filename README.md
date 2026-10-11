# Dự Án Nhận Diện Từ Khóa (Keyword Spotting - KWS) Cho ESP32-S3

Dự án phát triển mô hình TinyML Keyword Spotting (KWS) tiếng Việt offline, gọn nhẹ để triển khai trực tiếp trên chip vi điều khiển **ESP32-S3** theo quy trình chuẩn của **Edge Impulse**.

---

## 📁 Cấu Trúc Dự Án

- 📂 [dataset/](file:///d:/KWS/dataset): Chứa toàn bộ **3,543 files WAV** chuẩn hóa (16kHz, 16-bit PCM, Mono, 1.0s) gồm 10 lớp:
  - 8 từ khóa: `bật` (375), `tắt` (355), `đèn` (395), `quạt` (352), `phòng` (344), `khách` (336), `bếp` (355), `ngủ` (343).
  - 2 lớp chống kích hoạt nhầm: `noise` (360), `unknown` (328).
- 📂 [scripts/](file:///d:/KWS/scripts):
  - 📄 [preprocess_dataset.py](file:///d:/KWS/scripts/preprocess_dataset.py): Script làm sạch, loại bỏ file hỏng, resample và căn chỉnh thời lượng 1.0s.
  - 📄 [generate_noise_dataset.py](file:///d:/KWS/scripts/generate_noise_dataset.py): Script tải và tổng hợp dữ liệu tiếng ồn (ESC-50 + tạp âm phòng/phần cứng).
  - 📄 [generate_unknown_dataset.py](file:///d:/KWS/scripts/generate_unknown_dataset.py): Script tổng hợp 328 mẫu giọng nói tiếng Việt đa dạng ngoài 8 từ khóa.
- 📂 [firmware/](file:///d:/KWS/firmware): Dự án PlatformIO hoàn chỉnh cho **ESP32-S3**:
  - 📄 [platformio.ini](file:///d:/KWS/firmware/platformio.ini): Cấu hình board `esp32-s3-devkitc-1`, 240MHz, Flash 80MHz, cờ tối ưu `-O3`.
  - 📂 [include/](file:///d:/KWS/firmware/include):
    - [model_data.h](file:///d:/KWS/firmware/include/model_data.h): Mô hình lượng tử hóa INT8 (28 KB).
    - [kws_config.h](file:///d:/KWS/firmware/include/kws_config.h): Sơ đồ chân I2S (SD: GPIO 4, SCK: GPIO 5, WS: GPIO 6) & ngưỡng nhận diện.
    - [mfcc.h](file:///d:/KWS/firmware/include/mfcc.h): Khối trích xuất MFCC on-device thời gian thực.
  - 📂 [src/](file:///d:/KWS/firmware/src):
    - [main.cpp](file:///d:/KWS/firmware/src/main.cpp): Đọc I2S DMA, cửa sổ trượt 250ms, State Machine nhận diện chuỗi lệnh nhà thông minh.
    - [mfcc.cpp](file:///d:/KWS/firmware/src/mfcc.cpp): Giải thuật FFT 512 + Mel Filterbank + DCT-II tối ưu.

---

## 🚀 Trạng Thái Dự Án Hiện Tại
- ✅ **Giai đoạn 1 (Làm sạch & Chuẩn hóa Dataset)**: Đã hoàn thành 100% (3,543 files).
- ✅ **Giai đoạn 2 (Trích xuất MFCC DSP)**: Hoàn tất trích xuất đặc trưng `(3543, 99, 13)`.
- ✅ **Giai đoạn 3 & 4 (Huấn luyện & Lượng tử hóa INT8)**: Hoàn thành (Test Acc: **80.56%**, Rejection noise/unknown > 91-98%).
- ✅ **Giai đoạn 5 (Dự án Firmware PlatformIO cho ESP32-S3)**: **Đã khởi tạo và biên dịch thành công 100% [SUCCESS]**!
  - RAM sử dụng: **62.3%** Internal SRAM (không cần PSRAM).
  - Flash sử dụng: **9.0%** (301 KB).
