import sys
import time
import serial

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PORT = "COM10"
BAUD = 115200

def monitor(duration=10):
    print(f"Đang kết nối cổng {PORT} ở tốc độ {BAUD} baud...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1.0)
        ser.dtr = True
        ser.rts = True
        time.sleep(0.5)

        # Gửi ký tự newline để đánh thức Serial nếu cần
        ser.write(b"\n")

        print(f"--- BẮT ĐẦU THEO DÕI SERIAL TỪ ESP32-S3 ({PORT}) ---")
        start_time = time.time()
        while time.time() - start_time < duration:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='replace').rstrip()
                if line:
                    print(line)
            else:
                time.sleep(0.05)

        ser.close()
        print(f"--- KẾT THÚC THEO DÕI ({duration}s) ---")
    except Exception as e:
        print(f"Lỗi mở cổng {PORT}: {e}")

if __name__ == "__main__":
    dur = 8
    if len(sys.argv) > 1:
        try:
            dur = int(sys.argv[1])
        except:
            pass
    monitor(dur)
