# CÁC CHIẾN LƯỢC NHẬN DIỆN TỪ KHÓA LIÊN TỤC (STREAMING KWS) CHO ESP32-S3

Tài liệu này tổng hợp 2 phương pháp giải quyết bài toán: **Làm sao để người dùng có thể nói các từ khóa gần nhau hoặc nói liền mạch mà không cần ngắt quãng quá lâu.**

---

## 1. PHÂN TÍCH VẤN ĐỀ HIỆN TẠI

- **Dữ liệu huấn luyện**: Mô hình TinyKWS (DSCNN) được train trên các file WAV đơn lẻ 1.0s (16,000 mẫu), mỗi file chỉ chứa **duy nhất 1 từ khóa** nằm giữa khoảng lặng.
- **Hiện tượng thực tế**:
  - Âm tiết tiếng Việt (*"bật"*, *"tắt"*, *"đèn"*, *"quạt"*) chỉ kéo dài **250ms – 350ms**.
  - Khi người dùng nói *"bật đèn"* tự nhiên, toàn bộ 2 từ chỉ mất ~600ms – 750ms.
  - Cả 2 từ cùng lọt vào một khung 1.0s $\rightarrow$ Mạng nơ-ron nhận cả 2 từ cùng lúc và phân loại sai thành `noise`/`unknown`.
  - Cooldown 500ms cứng làm chip bỏ qua hoàn toàn từ thứ 2 nếu nói nhanh.

---

## 2. PHƯƠNG PHÁP 1: SLIDING WINDOW 100ms + POSTERIOR SMOOTHING

### 2.1. Nguyên lý hoạt động
- Giảm bước trượt (step size) từ **250ms xuống 100ms** (hoặc 120ms).
- Cửa sổ 1.0s trượt liên tục 10 lần mỗi giây.
- Sử dụng hàng đợi làm mịn xác suất (Probability Averaging Window) qua 3–4 khung (300ms–400ms).
- Khi xác suất của một từ khóa đạt đỉnh cực đại (Peak Detection) và vượt ngưỡng, ghi nhận từ đó và đặt cờ khóa từ trong 200ms để tránh lặp từ.

### 2.2. Đánh giá
- **Ưu điểm**:
  - Không cần bộ tách âm, theo dõi liên tục dòng âm thanh.
  - Tương thích tốt với các mô hình phát hiện từ kích hoạt đơn (Wake-word đơn lẻ như *"Alexa"*).
- **Nhược điểm trên ESP32-S3**:
  - Thời gian suy luận mạng DSCNN là **~88ms**.
  - Nếu bước trượt là 100ms, chip phải chạy suy luận liên tục 10 lần/giây $\rightarrow$ Chiếm **~90% CPU**, sinh nhiệt và dễ gây nghẽn buffer I2S DMA.
  - Vẫn có thể bị nhiễu nếu 2 từ quá sát nhau trong cùng 1 cửa sổ 1.0s.

---

## 3. PHƯƠNG PHÁP 2: VAD-BASED SYLLABLE SEGMENTATION & CENTERING (ĐỀ XUẤT ÁP DỤNG)

### 3.1. Nguyên lý hoạt động (Cắt & Căn giữa âm tiết độc lập)
Thay vì trượt một cửa sổ 1.0s cố định và đoán mò:
1. **Phát hiện bắt đầu nói (Speech Start)**:
   - Sử dụng bộ đệm vòng Pre-roll (~100ms) để không bị mất phụ âm đầu (*b*, *t*, *đ*).
   - Khi năng lượng âm thanh vượt ngưỡng nói $\rightarrow$ chuyển sang trạng thái gom âm (`VAD_SPEAKING`).
2. **Phát hiện kết thúc âm tiết (Speech End / Endpointing)**:
   - Trong tiếng Việt, giữa 2 từ *"bật"* và *"đèn"* luôn có một khoảng lặng vi mô tự nhiên (**Micro-silence ~80ms – 120ms**) do chuyển khẩu hình.
   - Khi năng lượng hạ thấp liên tục trong ~100ms $\rightarrow$ xác định người dùng đã nói xong 1 từ.
3. **Căn giữa & Đệm chuẩn 1.0s (Centering & Padding)**:
   - Lấy đúng đoạn âm tiết vừa nói (ví dụ 350ms).
   - Đặt đoạn âm tiết này vào **chính giữa buffer 1.0s** (16,000 mẫu), đệm 0 vào đầu và cuối:
     ```
     [0000... (325ms) | --- ÂM TIẾT NÓI (350ms) --- | 0000... (325ms)] = 1,000ms
     ```
   - **Đặc điểm vượt trội**: Khớp 100% với phân phối dữ liệu lúc train mô hình!
4. **Suy luận đúng 1 lần (Event-Driven Inference)**:
   - Chỉ chạy mô hình khi vừa nói xong từ $\rightarrow$ Tiết kiệm 85% CPU.
   - Trả kết quả ngay lập tức trong 88ms.
   - Sẵn sàng đón từ thứ hai ngay sau đó mà không cần bất kỳ cooldown cứng nào.

### 3.2. Bảng so sánh 2 phương pháp

| Tiêu chí | Cách 1: Sliding Window 100ms | Cách 2: VAD Syllable Segmentation |
| :--- | :---: | :---: |
| **Tải CPU ESP32-S3** | ~88% – 95% (liên tục) | **~10% – 15% (chỉ tính khi nói xong)** |
| **Độ khớp dữ liệu train** | Trung bình (từ bị lệch vị trí) | **Cực cao (từ được căn giữa chuẩn 1.0s)** |
| **Khả năng nói liền từ** | Cần khoảng nghỉ ~300ms | **Chỉ cần nghỉ tự nhiên ~80–120ms** |
| **Nguy cơ tràn DMA** | Cao (do CPU luôn bận) | **Rất thấp (CPU rảnh rỗi phần lớn thời gian)** |
| **Độ tin cậy nhận diện** | ~60% – 70% | **> 85% – 95%** |

---

## 4. KẾT LUẬN & HƯỚNG TRIỂN KHAI
Phương pháp 2 (**VAD-Based Syllable Segmentation**) là giải pháp tối ưu nhất cho ESP32-S3. Nó chuyển hệ thống từ cơ chế quét đồng hồ liên tục sang **cơ chế hướng sự kiện (Event-driven KWS)**, giúp nhận diện tự nhiên, nhanh nhạy và chính xác.
