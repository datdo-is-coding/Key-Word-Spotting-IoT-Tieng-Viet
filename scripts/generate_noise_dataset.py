import os
import sys
import io
import csv
import random
import requests
import numpy as np
import soundfile as sf
import scipy.signal

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "d:/KWS/dataset/noise"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_SAMPLES = 16000
TOTAL_TARGET = 360

def generate_silence_and_mic_floor(count):
    samples = []
    for _ in range(count):
        # Mức ồn nền micro phòng yên tĩnh (RMS rất nhỏ: 0.0001 - 0.002)
        amp = random.uniform(0.00005, 0.0015)
        noise = np.random.normal(0, amp, TARGET_SAMPLES).astype(np.float32)
        samples.append(noise)
    return samples

def generate_pink_noise(count):
    # Pink noise: 1/f noise
    samples = []
    for _ in range(count):
        white = np.random.normal(0, 1, TARGET_SAMPLES)
        # Bộ lọc IIR xấp xỉ pink noise (Paul Kellet's filter)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink = scipy.signal.lfilter(b, a, white)
        # Chuẩn hóa âm lượng ngẫu nhiên
        amp = random.uniform(0.01, 0.15)
        pink = pink / (np.max(np.abs(pink)) + 1e-6) * amp
        samples.append(pink.astype(np.float32))
    return samples

def generate_brown_noise(count):
    # Brown noise: tích phân của white noise (1/f^2)
    samples = []
    for _ in range(count):
        white = np.random.normal(0, 1, TARGET_SAMPLES)
        brown = np.cumsum(white)
        # Detrend
        brown = brown - np.mean(brown)
        amp = random.uniform(0.01, 0.15)
        brown = brown / (np.max(np.abs(brown)) + 1e-6) * amp
        samples.append(brown.astype(np.float32))
    return samples

def generate_white_noise(count):
    samples = []
    for _ in range(count):
        amp = random.uniform(0.005, 0.08)
        white = np.random.normal(0, amp, TARGET_SAMPLES).astype(np.float32)
        samples.append(white)
    return samples

def generate_mains_hum(count):
    # Tiếng ù điện lưới 50Hz hoặc 60Hz + các sóng hài bậc 2, 3, 4
    samples = []
    t = np.linspace(0, 1.0, TARGET_SAMPLES, endpoint=False)
    for _ in range(count):
        base_f = random.choice([50.0, 60.0])
        sig = 0.5 * np.sin(2 * np.pi * base_f * t)
        sig += 0.25 * np.sin(2 * np.pi * (2 * base_f) * t)
        sig += 0.12 * np.sin(2 * np.pi * (3 * base_f) * t)
        sig += 0.08 * np.sin(2 * np.pi * (4 * base_f) * t)
        # Thêm chút noise nền
        sig += np.random.normal(0, 0.05, TARGET_SAMPLES)
        amp = random.uniform(0.02, 0.12)
        sig = sig / (np.max(np.abs(sig)) + 1e-6) * amp
        samples.append(sig.astype(np.float32))
    return samples

def download_esc50_noise(target_slices=120):
    print("Đang tải danh sách metadata ESC-50 từ GitHub...")
    meta_url = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/meta/esc50.csv"
    res = requests.get(meta_url, timeout=10)
    if res.status_code != 200:
        print("Không tải được metadata ESC-50, bỏ qua phần này.")
        return []

    lines = res.text.splitlines()
    reader = csv.DictReader(lines)
    
    # Chọn các danh mục tiếng ồn thực tế trong nhà/môi trường
    target_cats = {
        'keyboard_typing', 'mouse_click', 'clock_tick', 'door_wood_knock',
        'vacuum_cleaner', 'washing_machine', 'wind', 'rain', 'water_drops',
        'footsteps', 'breathing'
    }
    candidates = [row['filename'] for row in reader if row['category'] in target_cats]
    random.shuffle(candidates)

    print(f"Tìm thấy {len(candidates)} file ESC-50 phù hợp. Đang tải một phần để cắt lát 1.0s...")
    slices = []
    download_count = 0
    base_url = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/"

    for fname in candidates:
        if len(slices) >= target_slices or download_count >= 30:
            break
        try:
            r = requests.get(base_url + fname, timeout=10)
            if r.status_code == 200:
                download_count += 1
                data, sr = sf.read(io.BytesIO(r.content), dtype='float32')
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if sr != TARGET_SR:
                    gcd = np.gcd(sr, TARGET_SR)
                    data = scipy.signal.resample_poly(data, TARGET_SR // gcd, sr // gcd)

                # Mỗi file ESC-50 dài 5 giây, cắt lấy các lát 1.0 giây
                num_sub = len(data) // TARGET_SAMPLES
                for s in range(num_sub):
                    if len(slices) >= target_slices:
                        break
                    sub_clip = data[s * TARGET_SAMPLES : (s + 1) * TARGET_SAMPLES]
                    # Điều chỉnh âm lượng hợp lý
                    max_v = np.max(np.abs(sub_clip))
                    if max_v > 0.005:
                        sub_clip = sub_clip / max_v * random.uniform(0.05, 0.4)
                        slices.append(sub_clip)
                print(f"  + Tải file {fname}: đã có {len(slices)} lát...")
        except Exception as e:
            print(f"Lỗi tải {fname}: {e}")

    return slices

def create_noise_dataset():
    print(f"Bắt đầu tổng hợp tập dữ liệu 'noise' (Mục tiêu: {TOTAL_TARGET} mẫu)...")
    all_samples = []

    # 1. Tải tiếng ồn môi trường thực tế từ ESC-50 (~120 mẫu)
    esc_slices = download_esc50_noise(target_slices=120)
    all_samples.extend(esc_slices)
    print(f"-> Đã có {len(esc_slices)} mẫu tiếng ồn môi trường ESC-50.")

    # 2. Sinh các loại ồn nền phần cứng & phòng
    silence = generate_silence_and_mic_floor(60)
    pink = generate_pink_noise(60)
    brown = generate_brown_noise(50)
    white = generate_white_noise(40)
    hum = generate_mains_hum(40)

    all_samples.extend(silence)
    all_samples.extend(pink)
    all_samples.extend(brown)
    all_samples.extend(white)
    all_samples.extend(hum)

    print(f"-> Đã sinh tổng cộng {len(all_samples)} mẫu noise. Bắt đầu lưu...")
    # Cắt hoặc lấy ngẫu nhiên đúng số lượng mục tiêu
    random.shuffle(all_samples)
    if len(all_samples) > TOTAL_TARGET:
        all_samples = all_samples[:TOTAL_TARGET]

    for idx, sample in enumerate(all_samples, 1):
        out_path = os.path.join(OUTPUT_DIR, f"{idx}.wav")
        # Đảm bảo độ dài đúng 16000
        if len(sample) != TARGET_SAMPLES:
            if len(sample) < TARGET_SAMPLES:
                sample = np.pad(sample, (0, TARGET_SAMPLES - len(sample)), 'constant')
            else:
                sample = sample[:TARGET_SAMPLES]
        sf.write(out_path, sample, TARGET_SR, subtype='PCM_16')

    print(f"Hoàn thành! Đã tạo {len(all_samples)} mẫu trong {OUTPUT_DIR}")

if __name__ == "__main__":
    create_noise_dataset()
