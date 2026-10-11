import sys
import time
import serial

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PORT = "COM10"
BAUD = 115200

def test_device():
    print(f"Kết nối tới {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1.0)
        ser.dtr = True
        ser.rts = True
        time.sleep(1.0)

        # 1. Gửi lệnh BẬT ĐÈN (Relay 1 - IO4)
        print("\n>>> Gửi lệnh: 'bat den'")
        ser.write(b"bat den\n")
        time.sleep(0.5)
        while ser.in_waiting > 0:
            print("ESP32:", ser.readline().decode('utf-8', errors='replace').rstrip())

        # 2. Gửi lệnh BẬT QUẠT (Relay 2 - IO5)
        time.sleep(1.0)
        print("\n>>> Gửi lệnh: 'bat quat'")
        ser.write(b"bat quat\n")
        time.sleep(0.5)
        while ser.in_waiting > 0:
            print("ESP32:", ser.readline().decode('utf-8', errors='replace').rstrip())

        # 3. Gửi lệnh TẮT ĐÈN
        time.sleep(1.0)
        print("\n>>> Gửi lệnh: 'tat den'")
        ser.write(b"tat den\n")
        time.sleep(0.5)
        while ser.in_waiting > 0:
            print("ESP32:", ser.readline().decode('utf-8', errors='replace').rstrip())

        # 4. Gửi lệnh TẮT QUẠT
        time.sleep(1.0)
        print("\n>>> Gửi lệnh: 'tat quat'")
        ser.write(b"tat quat\n")
        time.sleep(0.5)
        while ser.in_waiting > 0:
            print("ESP32:", ser.readline().decode('utf-8', errors='replace').rstrip())

        ser.close()
        print("\n[THÀNH CÔNG] Đã kiểm thử gửi/nhận lệnh qua cổng Serial COM10!")
    except Exception as e:
        print(f"Lỗi cổng Serial: {e}")

if __name__ == "__main__":
    test_device()
