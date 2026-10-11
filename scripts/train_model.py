import os
import sys
import json
import time
import numpy as np

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = "d:/KWS/data"
MODEL_DIR = "d:/KWS/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Định nghĩa kiến trúc mạng Tiny KWS Depthwise Separable CNN (DSCNN) cho ESP32-S3
class TinyKWS(nn.Module):
    def __init__(self, num_classes=10):
        super(TinyKWS, self).__init__()
        # Khối Conv 1 (Standard Conv2D)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout(0.15)

        # Khối Depthwise Separable Conv 2
        self.dw2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1, groups=16)
        self.pw2 = nn.Conv2d(16, 32, kernel_size=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout(0.2)

        # Khối Depthwise Separable Conv 3
        self.dw3 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1, groups=32)
        self.pw3 = nn.Conv2d(32, 64, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.drop3 = nn.Dropout(0.25)

        # Phân lớp
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.drop1(self.pool1(self.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.relu(self.bn2(self.pw2(self.dw2(x))))))
        x = self.drop3(self.pool3(self.relu(self.bn3(self.pw3(self.dw3(x))))))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Thiết bị huấn luyện: {device}")

    # 2. Đọc dữ liệu đặc trưng MFCC
    print("Đang nạp dữ liệu từ d:/KWS/data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with open(os.path.join(DATA_DIR, "labels.json"), "r", encoding="utf-8") as f:
        labels_meta = json.load(f)
    classes = labels_meta["classes"]
    num_classes = len(classes)

    # 3. Chuẩn hóa ma trận MFCC (Z-score normalization)
    mean = np.mean(X_train)
    std = np.std(X_train) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Lưu lại mean & std để phục vụ suy luận thực tế trên ESP32
    norm_params = {"mean": float(mean), "std": float(std)}
    with open(os.path.join(MODEL_DIR, "norm_params.json"), "w") as f:
        json.dump(norm_params, f, indent=2)

    # Thêm kênh channel (Batch, Channel=1, Time=99, Coefficients=13)
    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()), batch_size=64, shuffle=False)

    # 4. Khởi tạo mô hình
    model = TinyKWS(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Kiến trúc mô hình: TinyKWS (DSCNN)")
    print(f"Tổng số tham số: {total_params:,} tham số (~{total_params * 1 / 1024:.1f} KB dạng INT8)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    epochs = 45
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, "kws_dscnn_best.pt")

    print("\n" + "="*50)
    print("BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH KWS")
    print("="*50)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == epochs or val_acc > 0.95:
            print(f"Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss/train_total:.4f} - Acc: {train_acc*100:.2f}% | Val Loss: {val_loss/val_total:.4f} - Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nHuấn luyện hoàn tất trong {total_time:.1f} giây!")

    # 5. Đánh giá trên tập Test độc lập
    print("\n" + "="*50)
    print("ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP")
    print("="*50)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = np.mean(all_preds == all_targets)

    print(f"ĐỘ CHÍNH XÁC TẬP TEST (TEST ACCURACY): {test_acc * 100:.2f}%")
    print("\nChi tiết từng từ khóa:")
    for idx, cls_name in enumerate(classes):
        cls_mask = (all_targets == idx)
        if np.sum(cls_mask) > 0:
            cls_acc = np.mean(all_preds[cls_mask] == all_targets[cls_mask])
            print(f"  + Lớp '{cls_name:8s}': {cls_acc * 100:6.2f}% ({np.sum(cls_mask)} mẫu test)")

    # 6. Xuất mô hình sang định dạng ONNX phục vụ nạp vi điều khiển / Edge Impulse
    dummy_input = torch.randn(1, 1, 99, 13, device=device)
    onnx_path = os.path.join(MODEL_DIR, "kws_dscnn.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_audio_mfcc"],
        output_names=["probabilities"],
        opset_version=13
    )
    print(f"\nĐã xuất mô hình ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")

if __name__ == "__main__":
    train()
