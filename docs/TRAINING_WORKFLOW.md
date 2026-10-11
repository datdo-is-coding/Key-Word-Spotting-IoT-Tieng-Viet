# QUY TRÌNH HUẤN LUYỆN MÔ HÌNH KWS (KEYWORD SPOTTING) CHO ESP32-S3
*(Hướng dẫn chi tiết từng bước: Từ tiền xử lý, huấn luyện đến xuất mã nguồn C++ tương thích Edge Impulse)*

---

## 1. THIẾT LẬP MÔI TRƯỜNG PHÁT TRIỂN (ENVIRONMENT SETUP)

Bạn có thể thực hiện quy trình này hoàn toàn trên máy tính cá nhân bằng Python, hoặc kết hợp đẩy lên nền tảng đám mây **Edge Impulse Studio**.

### 1.1. Cài đặt các thư viện Python cần thiết
Mở terminal và cài đặt các gói xử lý âm thanh và Deep Learning:
```bash
pip install numpy scipy librosa soundfile matplotlib
pip install tensorflow
pip install edge-impulse-cli   # (Tùy chọn: Dùng nếu muốn đồng bộ lên Edge Impulse Studio)
```

---

## 2. BƯỚC 1: TIỀN XỬ LÝ & CHUẨN HÓA DỮ LIỆU (DATA CLEANING)

### 2.1. Yêu cầu chuẩn hóa
Mọi file âm thanh nạp vào mạng phải tuân thủ nghiêm ngặt chuẩn sau:
- **Sample Rate**: `16,000 Hz`
- **Số kênh**: `Mono` (1 channel)
- **Độ sâu bit**: `16-bit PCM`
- **Thời lượng**: Đúng `1.0 giây` (16,000 mẫu âm thanh).

### 2.2. Kịch bản tiền xử lý dữ liệu tự động (`scripts/preprocess_dataset.py`)
Script này sẽ làm 3 nhiệm vụ:
1. Phát hiện và xóa các file bị hỏng (như file `dataset/bếp/23.wav`).
2. Tự động chuyển đổi các file `48,000 Hz` thành `16,000 Hz`.
3. Chuẩn hóa độ dài: Cắt hoặc chèn thêm số 0 (Zero-padding) nếu file khác 1.0 giây.

```python
import os
import wave
import numpy as np
import soundfile as sf
import librosa

DATASET_DIR = "d:/KWS/dataset"
TARGET_SR = 16000
TARGET_SAMPLES = 16000 # 1.0 second @ 16kHz

def clean_and_resample():
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if not file.endswith(".wav"):
                continue
            path = os.path.join(root, file)
            try:
                # Kiểm tra dung lượng file
                if os.path.getsize(path) <= 44:
                    print(f"Xóa file lỗi rỗng: {path}")
                    os.remove(path)
                    continue

                # Đọc âm thanh và resample về 16kHz nếu cần
                audio, sr = librosa.load(path, sr=None, mono=True)
                if sr != TARGET_SR:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

                # Chuẩn hóa đúng 1.0 giây (16000 samples)
                if len(audio) < TARGET_SAMPLES:
                    audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), 'constant')
                elif len(audio) > TARGET_SAMPLES:
                    audio = audio[:TARGET_SAMPLES]

                # Lưu lại file chuẩn 16-bit PCM Mono 16kHz
                sf.write(path, audio, TARGET_SR, subtype='PCM_16')
            except Exception as e:
                print(f"Lỗi xử lý file {path}: {e}")
                if os.path.exists(path):
                    os.remove(path)

    print("Hoàn tất chuẩn hóa toàn bộ dataset về 16kHz 16-bit Mono!")

if __name__ == "__main__":
    clean_and_resample()
```

### 2.3. Bổ sung 2 lớp bắt buộc: `_noise` và `_unknown`
- **Lớp `_noise`**:
  - Tạo thư mục `dataset/_noise`.
  - Có thể dùng microphone thu 300–400 mẫu dài 1 giây gồm: tiếng quạt thổi, tiếng điều hòa, tiếng gõ phím, tiếng còi xe xa xăm, khoảng lặng trong phòng.
  - Hoặc trích xuất từ tập mã nguồn mở *Google Speech Commands Dataset* (thư mục `_background_noise_`).
- **Lớp `_unknown`**:
  - Tạo thư mục `dataset/_unknown`.
  - Cắt 400 đoạn 1 giây từ các cuộc hội thoại tiếng Việt thông thường (tin tức VTV, podcast, đọc truyện, các từ không nằm trong 8 từ khóa).

---

## 3. BƯỚC 2: QUY TRÌNH HUẤN LUYỆN (TRAINING PIPELINE)

Có 2 cách thực hiện huấn luyện: **Cách A (Edge Impulse Studio)** và **Cách B (Python Local)**. Cả 2 cách đều cho ra mô hình hoàn toàn tương thích với ESP32-S3.

---

### CÁCH A: SỬ DỤNG EDGE IMPULSE STUDIO (Khuyến nghị cho trực quan)

#### 1. Đẩy dữ liệu lên Edge Impulse
Sau khi cài đặt `edge-impulse-cli`, chạy lệnh:
```bash
# Đăng nhập vào Edge Impulse
edge-impulse-login

# Tải toàn bộ dataset lên với nhãn tự động theo tên thư mục
edge-impulse-uploader --category split d:/KWS/dataset/*/*.wav
```
*(Cờ `--category split` sẽ tự động phân chia 80% Train và 20% Test).*

#### 2. Cấu hình Impulse (Create Impulse)
- **Time series data**: Window size: `1000 ms`, Window increase: `200 ms`, Frequency: `16000 Hz`.
- **Processing block**: Chọn **Audio (MFCC)**.
- **Learning block**: Chọn **Classifier (Keras)**.
- Nhấn **Save Impulse**.

#### 3. Cấu hình DSP Block (MFCC)
- Number of coefficients: `13`
- Frame length: `0.02` (20 ms)
- Frame stride: `0.01` (10 ms)
- Filter number: `32`
- Nhấn **Save parameters** -> Nhấn **Generate features**.

#### 4. Huấn luyện mạng Classifier (NN Model)
- Cấu hình kiến trúc mạng:
  - Chọn kiến trúc: **1D Convolutional** hoặc **2D Convolutional** (với 2–3 lớp Conv, Dropout 0.25).
  - Number of training cycles (Epochs): `50`
  - Learning rate: `0.005`
  - Data augmentation: Bật **Add noise** và **Time-shift**.
- Nhấn **Start training**.

#### 5. Đánh giá & Xuất mã nguồn C++
- Chuyển sang tab **Deployment**.
- Chọn **C++ Library** (hoặc chọn **Espressif ESP-IDF component**).
- Tại phần **Optimization**, chọn **Quantized (int8)**.
- Nhấn **Build** -> Tải về file `.zip` chứa toàn bộ mã nguồn C/C++ đã được tối ưu hóa cho ESP-NN.

---

### CÁCH B: HUẤN LUYỆN TRỰC TIẾP TRÊN MÁY BẰNG PYTHON (OFFLINE)

Nếu bạn muốn kiểm soát hoàn toàn mã nguồn huấn luyện và tự động hóa trên máy nội bộ:

#### 1. Trích xuất đặc trưng MFCC (Khớp chuẩn Edge Impulse)
Sử dụng `librosa` hoặc `scipy.signal` để biến đổi tín hiệu 1D âm thanh thành ma trận 2D MFCC:
- Audio `16,000` samples $\rightarrow$ Ma trận kích thước `(99, 13, 1)`.

#### 2. Kiến trúc mạng DSCNN (Depthwise Separable CNN) tối ưu cho ESP32-S3
Mạng Depthwise Separable CNN giảm 70% số lượng phép tính so với CNN truyền thống mà vẫn giữ nguyên độ chính xác:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_kws_dscnn(input_shape=(99, 13, 1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Lớp Conv đầu tiên để trích xuất đặc trưng ban đầu
        layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Khối Depthwise Separable 1
        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, kernel_size=(1, 1)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),

        # Khối Depthwise Separable 2
        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, kernel_size=(1, 1)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Phân lớp
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```

#### 3. Lượng tử hóa Full INT8 (Full Integer Quantization)
Để chạy được trên **ESP-NN** của ESP32-S3 với tốc độ cao nhất:
```python
def convert_to_tflite_int8(saved_model_path, representative_dataset_gen, output_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    # Ép buộc toàn bộ phép toán chạy số nguyên INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"Đã xuất mô hình TFLite INT8: {output_path} ({len(tflite_quant_model) / 1024:.2f} KB)")
```

#### 4. Chuyển đổi sang file C Header (`model_data.h`)
Sử dụng công cụ `xxd` hoặc script Python để nhúng trực tiếp mô hình vào firmware ESP32:
```bash
xxd -i model_quantized.tflite > model_data.h
```

---

## 4. BƯỚC 3: TRIỂN KHAI TRÊN PHẦN CỨNG ESP32-S3

### 4.1. Sơ đồ kết nối phần cứng (ESP32-S3 + Microphone I2S INMP441)
| Chân Microphone (INMP441) | Chân ESP32-S3 (GPIO khuyến nghị) | Mô tả tín hiệu |
| :--- | :--- | :--- |
| **VDD** | 3.3V | Nguồn cấp |
| **GND** | GND | Nối đất |
| **SD** | GPIO 4 | I2S Serial Data Out |
| **SCK** (BCLK) | GPIO 5 | I2S Bit Clock |
| **WS** (LRCK) | GPIO 6 | I2S Word Select (Left/Right) |
| **L/R** | GND | Kênh Left (Mono) |

### 4.2. Nguyên lý nhận diện thời gian thực (Continuous Inference)
1. **DMA Double Buffering**: I2S thu âm liên tục không bị gián đoạn vào bộ nhớ đệm DMA.
2. **Sliding Window (Cửa sổ trượt)**:
   - Thay vì chờ đủ 1 giây mới đoán 1 lần (gây trễ và dễ đứt câu), ESP32-S3 lưu một mảng trượt 16,000 mẫu.
   - Cứ mỗi **250 ms**, hệ thống cập nhật 4,000 mẫu âm thanh mới và đẩy qua bộ trích xuất MFCC $\rightarrow$ Suy luận $\rightarrow$ Trả kết quả.
3. **Threshold & Cooldown Filter**:
   - Chỉ kích hoạt khi xác suất `P(từ khóa) > 0.85`.
   - Bật cờ Cooldown khoảng 500ms sau khi nhận diện thành công để tránh nhận diện trùng lặp 1 từ 2 lần liên tiếp.

---

## 5. BẢNG CHECKLIST KIỂM THỬ TRƯỚC KHI DEPLOY

- [ ] Toàn bộ audio đã được chuẩn hóa về **16,000 Hz, Mono, 16-bit**.
- [ ] File hỏng `bếp/23.wav` đã được loại bỏ.
- [ ] Tập dữ liệu có đủ 10 lớp (`8 từ khóa + _noise + _unknown`).
- [ ] Độ chính xác trên tập Test đạt $\ge 93\%$.
- [ ] Dung lượng file `.tflite` INT8 $\le 100\text{ KB}$.
- [ ] Thời gian suy luận trên ESP32-S3 $\le 30\text{ ms}$.
- [ ] Không có hiện tượng kích hoạt giả (False Trigger) khi để micro trong phòng yên tĩnh 30 phút.
