#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
import serial
import time
import binascii

# Command to give the permission
#sudo chmod 777 /dev/ttyUSB0

# Lần lượt thay đổi 1 trong 3 sẽ có cái chạy ok
#port_ = '/dev/ttyUSB0'
port_ = '/dev/ttyUSB1'
#port_ = '/dev/ttyUSB2'

ser = serial.Serial(port='COM12',baudrate=115200,timeout=1, parity=serial.PARITY_NONE,
stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS)
counter = 0

# while counter < 1:
#    counter = counter + 1
#    #ser.write("\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00\x73\x30")
#    ser.write(serial.to_bytes([0x09,0x10,0x03,0xE8,0x00,0x03,0x06,0x00,0x00,0x00,0x00,0x00,0x00,0x73,0x30]))


#    data_raw = ser.readline()
#    print(data_raw)
#    data = binascii.hexlify(data_raw)
#    print ("Response 1 ", data)
#    time.sleep(0.01)
 
#    #ser.write("\x09\x03\x07\xD0\x00\x01\x85\xCF")
#    ser.write(serial.to_bytes([0x09,0x03,0x07,0xD0,0x00,0x01,0x85,0xCF]))

#    data_raw = ser.readline()
#    print(data_raw)
#    data = binascii.hexlify(data_raw)
#    print ("Response 2 ", data)
#    time.sleep(1)
ser.write(serial.to_bytes([0x09,0x10,0x03,0xE8,0x00,0x03,0x06,0x00,0x00,0x00,0x00,0x00,0x00,0x73,0x30]))
while(True):
   print ("Close gripper")
   #ser.write("\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29")
   #ser.write(serial.to_bytes([0x09,0x10,0x03,0xE8,0x00,0x03,0x06,0x09,0x00,0x00,0xFF,0xFF,0xFF,0x42,0x29])) # Full speed & Full force & close 100%
   #ser.write(serial.to_bytes([0x09,0x10,0x03,0xE8,0x00,0x03,0x06,0x09,0x00,0x00,0xBF,0xFF,0xFF,0x43,0xFD])) # Full speed & Full force & close 75%
   #ser.write(serial.to_bytes([0x09,0x10,0x03,0xE8,0x00,0x03,0x06,0x09,0x00,0x00,0x7F,0x7F,0x7F,0x23,0xA1])) # Half speed & Half force & close 50%
   ser.write(serial.to_bytes([0x09,0x10,0x03,0xE8,0x00,0x03,0x06,0x09,0x00,0x00,0x3F,0x3F,0x3F,0x12,0x45])) # 1/4 speed & 1/4 force & close 25%

   data_raw = ser.readline()
   print(data_raw)
   data = binascii.hexlify(data_raw)
   print ("Response 3 ", data)
   time.sleep(2)
 
   print ("Open gripper")
   #ser.write("\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\xFF\xFF\x72\x19")
   ser.write(serial.to_bytes([0x09,0x10,0x03,0xE8,0x00,0x03,0x06,0x09,0x00,0x00,0x00,0xFF,0xFF,0x72,0x19]))# Full speed & Full force & open 100%

   data_raw = ser.readline()
   print(data_raw)
   data = binascii.hexlify(data_raw)
   print ("Response 4 ", data)
   time.sleep(2)
