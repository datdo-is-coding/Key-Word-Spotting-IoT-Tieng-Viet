import os
import sys

def convert_file_to_c_header(input_bin_path, output_header_path, array_name="kws_model_data"):
    """Chuyển đổi bất kỳ file model nhị phân (.onnx / .tflite / .pt) thành mảng C const unsigned char"""
    if not os.path.exists(input_bin_path):
        print(f"File không tồn tại: {input_bin_path}")
        return

    with open(input_bin_path, "rb") as f:
        data = f.read()

    size = len(data)
    print(f"Chuyển đổi {input_bin_path} ({size:,} bytes) thành {output_header_path}...")

    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write("// Tự động tạo bởi script KWS export cho ESP32-S3\n")
        f.write(f"// Kích thước mô hình: {size:,} bytes ({size / 1024:.2f} KB)\n\n")
        f.write("#ifndef KWS_MODEL_DATA_H\n")
        f.write("#define KWS_MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const unsigned int {array_name}_len = {size};\n")
        f.write(f"const unsigned char {array_name}[] = {{\n  ")

        for i, byte in enumerate(data):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 16 == 0:
                f.write("\n  ")

        f.write("\n};\n\n")
        f.write("#endif // KWS_MODEL_DATA_H\n")

    print(f"Đã xuất thành công: {output_header_path}")

if __name__ == "__main__":
    onnx_file = "d:/KWS/models/kws_dscnn.onnx"
    header_file = "d:/KWS/models/model_data.h"
    if os.path.exists(onnx_file):
        convert_file_to_c_header(onnx_file, header_file)
    else:
        print(f"Chờ mô hình ONNX hoàn thành huấn luyện...")
