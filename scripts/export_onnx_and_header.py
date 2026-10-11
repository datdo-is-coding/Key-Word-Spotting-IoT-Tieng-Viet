import os
import sys
import torch
import torch.nn as nn
from train_model import TinyKWS
from export_c_header import convert_file_to_c_header

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_DIR = "d:/KWS/models"
pt_path = os.path.join(MODEL_DIR, "kws_dscnn_best.pt")
onnx_path = os.path.join(MODEL_DIR, "kws_dscnn.onnx")
header_path = os.path.join(MODEL_DIR, "model_data.h")

def export_all():
    print(f"Đang nạp trọng số từ {pt_path}...")
    model = TinyKWS(num_classes=10)
    model.load_state_dict(torch.load(pt_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 1, 99, 13)
    print("Xuất sang ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_audio_mfcc"],
        output_names=["probabilities"],
        dynamic_axes={"input_audio_mfcc": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
        opset_version=13
    )
    print(f"-> File ONNX: {onnx_path} ({os.path.getsize(onnx_path) / 1024:.2f} KB)")

    print("Xuất sang file C Header nhúng trực tiếp ESP32-S3...")
    convert_file_to_c_header(onnx_path, header_path, array_name="kws_onnx_model")
    print("\nHOÀN TẤT XUẤT TOÀN BỘ MÔ HÌNH VÀ FILE C HEADER CHO ESP32-S3!")

if __name__ == "__main__":
    export_all()
