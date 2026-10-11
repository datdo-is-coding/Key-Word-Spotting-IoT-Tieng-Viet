import os
import sys
import json
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader, QuantType, QuantFormat

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from export_c_header import convert_file_to_c_header

MODEL_DIR = "d:/KWS/models"
DATA_DIR = "d:/KWS/data"

FP32_MODEL_PATH = os.path.join(MODEL_DIR, "kws_dscnn.onnx")
INT8_MODEL_PATH = os.path.join(MODEL_DIR, "kws_dscnn_int8.onnx")
INT8_HEADER_PATH = os.path.join(MODEL_DIR, "model_data_int8.h")

# Calibration Data Reader cho quá trình Static Quantization (PTQ)
class KWSCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_data, input_name="input_audio_mfcc", batch_size=1):
        self.data = calibration_data
        self.input_name = input_name
        self.batch_size = batch_size
        self.current_idx = 0

    def get_next(self):
        if self.current_idx >= len(self.data):
            return None
        batch = self.data[self.current_idx : self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        return {self.input_name: batch}

    def rewind(self):
        self.current_idx = 0

def run_quantization():
    print("="*60)
    print("BẮT ĐẦU QUÁ TRÌNH LƯỢNG TỬ HÓA INT8 (POST-TRAINING QUANTIZATION)")
    print("="*60)

    # 1. Đọc dữ liệu kiểm tra và tham số chuẩn hóa
    print("\n1. Đang nạp tập dữ liệu kiểm chuẩn (Calibration) và tập Test...")
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with open(os.path.join(MODEL_DIR, "norm_params.json")) as f:
        norm = json.load(f)
    mean, std = norm["mean"], norm["std"]

    with open(os.path.join(DATA_DIR, "labels.json"), encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"]

    X_val = (X_val - mean) / std
    X_val = np.expand_dims(X_val, 1).astype(np.float32)

    X_test = (X_test - mean) / std
    X_test = np.expand_dims(X_test, 1).astype(np.float32)

    # 2. Thực hiện Lượng tử hóa INT8
    print(f"\n2. Thực hiện Static Quantization (QDQ) trên file: {FP32_MODEL_PATH}...")
    calib_reader = KWSCalibrationDataReader(X_val, input_name="input_audio_mfcc", batch_size=1)

    try:
        quantize_static(
            model_input=FP32_MODEL_PATH,
            model_output=INT8_MODEL_PATH,
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True
        )
        print(f"-> Tạo thành công mô hình Static INT8: {INT8_MODEL_PATH}")
    except Exception as e:
        print(f"Lưu ý Static Quantization: {e}. Chuyển sang Dynamic/Weight-only Quantization...")
        quantize_dynamic(
            model_input=FP32_MODEL_PATH,
            model_output=INT8_MODEL_PATH,
            weight_type=QuantType.QInt8
        )
        print(f"-> Tạo thành công mô hình Dynamic INT8: {INT8_MODEL_PATH}")

    # 3. So sánh kích thước file
    fp32_size = os.path.getsize(FP32_MODEL_PATH)
    int8_size = os.path.getsize(INT8_MODEL_PATH)
    print("\n3. SO SÁNH DUNG LƯỢNG MÔ HÌNH:")
    print(f"  + Mô hình Float32 gốc: {fp32_size:,} bytes ({fp32_size/1024:.2f} KB)")
    print(f"  + Mô hình INT8 sau lượng tử: {int8_size:,} bytes ({int8_size/1024:.2f} KB)")
    print(f"  -> Giảm: {(1 - int8_size/fp32_size)*100:.1f}% dung lượng!")

    # 4. Đo đạc độ chính xác và độ trễ trên tập Test
    print("\n4. ĐO ĐẠC HIỆU NĂNG TRÊN TẬP TEST ĐỘC LẬP (355 MẪU):")

    # A. Đo Float32
    sess_fp32 = ort.InferenceSession(FP32_MODEL_PATH)
    t0 = time.time()
    preds_fp32 = []
    for i in range(len(X_test)):
        out = sess_fp32.run(None, {"input_audio_mfcc": X_test[i:i+1]})[0]
        preds_fp32.append(np.argmax(out))
    t_fp32 = (time.time() - t0) / len(X_test) * 1000  # ms per sample
    acc_fp32 = np.mean(np.array(preds_fp32) == y_test)

    # B. Đo INT8
    sess_int8 = ort.InferenceSession(INT8_MODEL_PATH)
    t0 = time.time()
    preds_int8 = []
    for i in range(len(X_test)):
        out = sess_int8.run(None, {"input_audio_mfcc": X_test[i:i+1]})[0]
        preds_int8.append(np.argmax(out))
    t_int8 = (time.time() - t0) / len(X_test) * 1000  # ms per sample
    acc_int8 = np.mean(np.array(preds_int8) == y_test)

    print("-" * 65)
    print(f"{'Chỉ số':<25} | {'Float32 (Gốc)':<18} | {'INT8 (Quantized)':<18}")
    print("-" * 65)
    print(f"{'Dung lượng Model':<25} | {fp32_size/1024:>12.2f} KB   | {int8_size/1024:>12.2f} KB")
    print(f"{'Độ chính xác Test':<25} | {acc_fp32*100:>14.2f}%   | {acc_int8*100:>14.2f}%")
    print(f"{'Thời gian suy luận':<25} | {t_fp32:>12.2f} ms/mẫu | {t_int8:>12.2f} ms/mẫu")
    print(f"{'Độ suy hao chính xác':<25} | {'-':>18} | {(acc_fp32 - acc_int8)*100:>13.2f}%")
    print("-" * 65)

    # 5. Chi tiết từng từ khóa sau Quantization
    print("\n5. CHI TIẾT ĐỘ CHÍNH XÁC TỪNG TỪ KHÓA SAU LƯỢNG TỬ HÓA:")
    preds_int8 = np.array(preds_int8)
    for idx, cls_name in enumerate(classes):
        mask = (y_test == idx)
        if np.sum(mask) > 0:
            c_fp32 = np.mean(np.array(preds_fp32)[mask] == y_test[mask]) * 100
            c_int8 = np.mean(preds_int8[mask] == y_test[mask]) * 100
            print(f"  + Lớp '{cls_name:<8s}': Float32 = {c_fp32:5.1f}%  -->  INT8 = {c_int8:5.1f}%")

    # 6. Xuất mã nguồn C Header INT8 cho ESP32-S3
    print("\n6. Đang xuất file C Header nhúng cho ESP32-S3...")
    convert_file_to_c_header(INT8_MODEL_PATH, INT8_HEADER_PATH, array_name="kws_int8_model")

    # Đồng thời cập nhật file model_data.h chính
    convert_file_to_c_header(INT8_MODEL_PATH, os.path.join(MODEL_DIR, "model_data.h"), array_name="kws_model")

    print(f"\n-> Đã cập nhật file C Header: {INT8_HEADER_PATH}")
    print(f"-> Đã cập nhật file C Header chính: {os.path.join(MODEL_DIR, 'model_data.h')}")

    # 7. Lưu báo cáo so sánh ra file JSON
    quant_report = {
        "fp32_size_kb": round(fp32_size / 1024, 2),
        "int8_size_kb": round(int8_size / 1024, 2),
        "fp32_accuracy": round(acc_fp32 * 100, 2),
        "int8_accuracy": round(acc_int8 * 100, 2),
        "accuracy_drop": round((acc_fp32 - acc_int8) * 100, 2),
        "fp32_latency_ms": round(t_fp32, 2),
        "int8_latency_ms": round(t_int8, 2),
        "per_class": {}
    }
    for idx, cls_name in enumerate(classes):
        mask = (y_test == idx)
        if np.sum(mask) > 0:
            quant_report["per_class"][cls_name] = {
                "fp32_acc": round(float(np.mean(np.array(preds_fp32)[mask] == y_test[mask]) * 100), 2),
                "int8_acc": round(float(np.mean(preds_int8[mask] == y_test[mask]) * 100), 2)
            }

    with open(os.path.join(MODEL_DIR, "quantization_report.json"), "w", encoding="utf-8") as f:
        json.dump(quant_report, f, indent=2, ensure_ascii=False)
    print(f"-> Đã lưu báo cáo định lượng: {os.path.join(MODEL_DIR, 'quantization_report.json')}")

    print("\nHOÀN TẤT LƯỢNG TỬ HÓA INT8 THÀNH CÔNG!")

if __name__ == "__main__":
    run_quantization()
