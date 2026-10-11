import os
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import json
import time
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg') # Chạy chế độ headless lưu ảnh
import matplotlib.pyplot as plt

# Cấu hình font chữ hiển thị chuẩn tiếng Việt trên Windows
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = "d:/KWS/data"
MODEL_DIR = "d:/KWS/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset tùy biến có hỗ trợ Data Augmentation (Time Shift & Additive Noise)
class KWSDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            # 1. Time shifting (dịch ngẫu nhiên dọc theo trục thời gian ±5 frames)
            shift = np.random.randint(-5, 6)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=1)
            # 2. Additive Gaussian noise nhẹ (SNR ngẫu nhiên)
            if np.random.rand() > 0.5:
                noise = torch.randn_like(x) * 0.05
                x = x + noise

        return x, y

# Kiến trúc mô hình TinyKWS (Depthwise Separable CNN)
class TinyKWS(nn.Module):
    def __init__(self, num_classes=10):
        super(TinyKWS, self).__init__()
        # Khối Conv 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout(0.15)

        # Khối Depthwise Separable 2
        self.dw2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1, groups=16)
        self.pw2 = nn.Conv2d(16, 32, kernel_size=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout(0.2)

        # Khối Depthwise Separable 3
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

def plot_and_save_visualizations(history, y_true, y_pred, classes):
    print("\nĐang tạo các biểu đồ trực quan hóa kết quả...")

    # 1. BIỂU ĐỒ 1: LOSS & ACCURACY QUA CÁC EPOCH
    plt.figure(figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'b-o', markersize=3, label='Train Loss')
    plt.plot(epochs, history["val_loss"], 'r-s', markersize=3, label='Validation Loss')
    plt.title('Hàm mất mát (Loss) qua từng Epoch', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], 'b-o', markersize=3, label='Train Accuracy')
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], 'r-s', markersize=3, label='Val Accuracy')
    plt.title('Độ chính xác (Accuracy %) qua từng Epoch', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Độ chính xác (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True)

    plt.tight_layout()
    loss_acc_path = os.path.join(MODEL_DIR, "training_loss_accuracy.png")
    plt.savefig(loss_acc_path, dpi=200)
    plt.close()
    print(f"-> Đã lưu biểu đồ Loss & Accuracy: {loss_acc_path}")

    # 2. BIỂU ĐỒ 2: MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix) Trên Tập Test', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar(label='Tỷ lệ dự đoán đúng')

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_str = f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)"
            plt.text(j, i, text_str,
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=9,
                     color="white" if cm_norm[i, j] > thresh else "black")

    plt.ylabel('Nhãn thực tế (True Label)', fontsize=12, fontweight='bold')
    plt.xlabel('Nhãn dự đoán (Predicted Label)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"-> Đã lưu Ma trận nhầm lẫn: {cm_path}")

    # 3. BIỂU ĐỒ 3: PRECISION, RECALL, F1-SCORE CHO TỪNG TỪ KHÓA
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    precisions = [report[cls]["precision"] * 100 for cls in classes]
    recalls = [report[cls]["recall"] * 100 for cls in classes]
    f1s = [report[cls]["f1-score"] * 100 for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(13, 6))
    plt.bar(x - width, precisions, width, label='Precision', color='#3498db')
    plt.bar(x, recalls, width, label='Recall', color='#2ecc71')
    plt.bar(x + width, f1s, width, label='F1-Score', color='#e67e22')

    plt.title('Đánh Giá Chi Tiết Theo Từng Lớp (Precision - Recall - F1)', fontsize=14, fontweight='bold')
    plt.xlabel('Từ Khóa / Lớp', fontsize=12)
    plt.ylabel('Tỷ lệ (%)', fontsize=12)
    plt.xticks(x, classes, fontsize=11)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(frameon=True, loc='lower right')
    plt.tight_layout()

    metrics_path = os.path.join(MODEL_DIR, "per_class_metrics.png")
    plt.savefig(metrics_path, dpi=200)
    plt.close()
    print(f"-> Đã lưu biểu đồ Precision/Recall/F1: {metrics_path}")

    # 4. TỔNG HỢP DASHBOARD 4-TRONG-1
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel A: Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history["val_loss"], 'r--', label='Val Loss', linewidth=2)
    ax1.set_title('(A) Đường Cong Hàm Mất Mát (Loss)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Panel B: Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], 'r--', label='Val Acc', linewidth=2)
    ax2.set_title('(B) Đường Cong Độ Chính Xác (Accuracy %)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # Panel C: Confusion Matrix
    ax3 = fig.add_subplot(gs[1, 0])
    im = ax3.imshow(cm_norm, cmap=plt.cm.Blues)
    ax3.set_title('(C) Ma Trận Nhầm Lẫn (Confusion Matrix)', fontsize=12, fontweight='bold')
    ax3.set_xticks(tick_marks)
    ax3.set_xticklabels(classes, rotation=45)
    ax3.set_yticks(tick_marks)
    ax3.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, f"{cm_norm[i, j]*100:.0f}%", ha="center", va="center",
                     fontsize=8, color="white" if cm_norm[i, j] > thresh else "black")

    # Panel D: F1-Score Bar Chart
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.bar(classes, f1s, color=['#2ecc71' if f >= 80 else '#e74c3c' for f in f1s])
    ax4.axhline(y=80, color='gray', linestyle='--', label='Ngưỡng 80%')
    ax4.set_title('(D) Điểm F1-Score Từng Từ Khóa', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1-Score (%)')
    ax4.set_ylim(0, 105)
    ax4.set_xticklabels(classes, rotation=45)
    ax4.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha='center', fontsize=8)

    dashboard_path = os.path.join(MODEL_DIR, "kws_evaluation_dashboard.png")
    fig.suptitle('BẢNG ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH KWS ESP32-S3', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(dashboard_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"-> Đã lưu Bảng Dashboard hoàn chỉnh: {dashboard_path}")

    return {
        "loss_acc_plot": loss_acc_path,
        "confusion_matrix_plot": cm_path,
        "metrics_plot": metrics_path,
        "dashboard_plot": dashboard_path,
        "classification_report": report
    }

def train_and_visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Thiết bị tính toán: {device}")

    # Đọc dữ liệu
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with open(os.path.join(DATA_DIR, "labels.json"), "r", encoding="utf-8") as f:
        labels_meta = json.load(f)
    classes = labels_meta["classes"]

    mean = np.mean(X_train)
    std = np.std(X_train) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)

    train_dataset = KWSDataset(X_train, y_train, augment=True)
    val_dataset = KWSDataset(X_val, y_val, augment=False)
    test_dataset = KWSDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = TinyKWS(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    epochs = 45
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, "kws_dscnn_best.pt")

    print("\nBắt đầu huấn luyện mô hình và ghi lại lịch sử từng epoch...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
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
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs:2d}] | Loss: {epoch_train_loss:.4f} / Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_train_acc*100:.2f}% / Val Acc: {epoch_val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

    print(f"Huấn luyện xong trong {time.time() - start_time:.1f}s.")

    # Đánh giá trên tập test
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(targets.numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Lưu lại lịch sử vào file json
    with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Xuất các biểu đồ trực quan hóa
    vis_res = plot_and_save_visualizations(history, y_true, y_pred, classes)

    # Copy ảnh sang thư mục artifacts để hiển thị cho người dùng
    artifact_dir = r"C:\Users\Admin\.gemini\antigravity-ide\brain\e905d283-4a08-4e27-afa6-ee04742952cb"
    if os.path.exists(artifact_dir):
        for img_name in ["kws_evaluation_dashboard.png", "training_loss_accuracy.png", "confusion_matrix.png", "per_class_metrics.png"]:
            src = os.path.join(MODEL_DIR, img_name)
            dst = os.path.join(artifact_dir, img_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"-> Đã sao chép {img_name} sang artifact: {dst}")

    print("\nHOÀN THÀNH TOÀN BỘ QUÁ TRÌNH TRỰC QUAN HÓA!")

if __name__ == "__main__":
    train_and_visualize()
