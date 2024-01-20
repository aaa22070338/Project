import serial
import time

ser = serial.Serial("COM15", 9600, timeout=1)
# while True:
#     val = ser.write("1".encode("utf-8"))
#     print("1")
time.sleep(1.4)
val = ser.write("0".encode("utf-8"))
print(ser)
time.sleep(1)   