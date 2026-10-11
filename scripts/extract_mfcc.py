import os
import sys
import glob
import json
import numpy as np
import soundfile as sf
import scipy.fftpack

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = "d:/KWS/dataset"
OUTPUT_DIR = "d:/KWS/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cấu hình chuẩn MFCC tương thích 100% với Edge Impulse DSP Block
SAMPLE_RATE = 16000
FRAME_LENGTH = 0.02   # 20ms = 320 samples
FRAME_STRIDE = 0.01   # 10ms = 160 samples
NUM_CEPSTRAL = 13     # 13 hệ số MFCC
NUM_FILTERS = 32      # 32 bộ lọc Mel
N_FFT = 512           # Kích thước FFT
LOW_FREQ = 300        # Tần số cắt dưới (Hz)
HIGH_FREQ = 8000      # Tần số cắt trên (Nyquist)
PRE_EMPHASIS = 0.97   # Bộ lọc tăng cường âm tần cao

def get_mel_filterbank(num_filters, n_fft, sample_rate, low_freq, high_freq):
    """Tạo ma trận bộ lọc tam giác Mel (Mel Filterbank)"""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((num_filters, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (f_m_plus - f_m)

    return fbank

MEL_FBANK = get_mel_filterbank(NUM_FILTERS, N_FFT, SAMPLE_RATE, LOW_FREQ, HIGH_FREQ)
HAMMING_WINDOW = np.hamming(int(round(FRAME_LENGTH * SAMPLE_RATE)))

def compute_mfcc(signal):
    """Trích xuất ma trận MFCC kích thước (99, 13) từ tín hiệu âm thanh 1D 16,000 samples"""
    # 1. Pre-emphasis
    emphasized = np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1])

    # 2. Phân khung (Framing)
    frame_len = int(round(FRAME_LENGTH * SAMPLE_RATE))
    frame_step = int(round(FRAME_STRIDE * SAMPLE_RATE))
    num_frames = int(np.ceil(float(np.abs(len(emphasized) - frame_len)) / frame_step)) + 1
    
    # Pad thêm nếu thiếu
    pad_len = (num_frames - 1) * frame_step + frame_len - len(emphasized)
    if pad_len > 0:
        emphasized = np.pad(emphasized, (0, pad_len), mode='constant')

    # Trích các khung dạng ma trận
    frames = np.lib.stride_tricks.sliding_window_view(emphasized, frame_len)[::frame_step]

    # 3. Nhân cửa sổ Hamming
    frames_win = frames * HAMMING_WINDOW

    # 4. FFT & Phổ công suất (Power Spectrum)
    mag_frames = np.absolute(np.fft.rfft(frames_win, N_FFT))
    pow_frames = (1.0 / N_FFT) * (mag_frames ** 2)

    # 5. Lọc qua dải Mel (Mel Filterbank)
    filter_banks = np.dot(pow_frames, MEL_FBANK.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_mel = np.log(filter_banks)

    # 6. Biến đổi Cosine rời rạc (DCT type 2) để lấy các hệ số MFCC
    mfcc = scipy.fftpack.dct(log_mel, type=2, axis=1, norm='ortho')[:, :NUM_CEPSTRAL]

    # Chuẩn hóa Cepstral liftering nhẹ (giống Edge Impulse)
    cep_lifter = 22
    n = np.arange(NUM_CEPSTRAL)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc = mfcc * lift

    return mfcc.astype(np.float32)

def extract_all():
    # 1. Xác định danh sách nhãn
    classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    label_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    idx_to_label = {idx: cls_name for idx, cls_name in enumerate(classes)}

    print(f"Các lớp nhận diện ({len(classes)} classes): {label_to_idx}")
    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, ensure_ascii=False, indent=2)

    all_features = []
    all_labels = []

    print("\nBắt đầu trích xuất MFCC features từ toàn bộ dataset...")
    total_processed = 0

    for cls_name in classes:
        cls_dir = os.path.join(DATASET_DIR, cls_name)
        wav_files = glob.glob(os.path.join(cls_dir, "*.wav"))
        label_idx = label_to_idx[cls_name]
        print(f"  -> Xử lý lớp '{cls_name}' ({len(wav_files)} files)...")

        for wav_path in wav_files:
            try:
                audio, sr = sf.read(wav_path, dtype='float32')
                if len(audio) != 16000:
                    if len(audio) < 16000:
                        audio = np.pad(audio, (0, 16000 - len(audio)), 'constant')
                    else:
                        audio = audio[:16000]

                feat = compute_mfcc(audio)
                all_features.append(feat)
                all_labels.append(label_idx)
                total_processed += 1
            except Exception as e:
                print(f"Lỗi file {wav_path}: {e}")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    print(f"\nTổng mẫu trích xuất thành công: {len(X)}")
    print(f"Kích thước ma trận đặc trưng X: {X.shape} (Mẫu, Frames, Coefficients)")
    print(f"Kích thước nhãn y: {y.shape}")

    # 2. Xáo trộn và phân chia tập dữ liệu: 80% Train, 10% Val, 10% Test
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    n_total = len(X)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    print(f"\nPhân chia dữ liệu:")
    print(f"  + Tập Train: {X_train.shape[0]} mẫu")
    print(f"  + Tập Validation: {X_val.shape[0]} mẫu")
    print(f"  + Tập Test: {X_test.shape[0]} mẫu")

    # 3. Lưu thành các file .npy nén
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"\nĐã lưu toàn bộ đặc trưng vào thư mục: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_all()
