import re
import socket
import serial
import time
import binascii


def robot_move(position, c:socket.socket):
    x, y, z, rx, ry, rz = position
    mesg = f'{x},{y},{z},{rx},{ry},{rz}'
    c.send(bytes(mesg, "utf-8"))
def receive(c:socket.socket):
    data = c.recv(1024).decode()
    data = re.sub(',', ' ', data)
    pos = data.split()
    pos = list(map(float, pos))  # [x, y, z, rx - psi, ry - theta, rz - phi]
    return pos
def crc_calculate(pos, velocity, acceleration):
    xor_constant = 0b1010000000000001
    is_first_value = True
    for input_value in [pos, velocity, acceleration]:
        
        if is_first_value:
            current_value = int(format(input_value ^ 0x79D1, "0>16b"), 2)
            is_first_value = False
        else:
            current_value = int(format(input_value ^ current_value, "0>16b"), 2)

        for i in range(8):
            prev_value = current_value
            current_value = current_value >> 1
            if (bin(prev_value)[-1]) == "1":
                current_value = int(format(current_value ^ xor_constant, "0>16b"), 2)
    high_byte = int(hex(current_value)[-2:], 16)
    low_byte = int(hex(current_value)[2:4], 16)
    return high_byte, low_byte

def gripMove(postion,velocity,force,COM,sleep_time):
    crc = crc_calculate(postion,velocity,force)
    activation_request = serial.to_bytes(
        [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x73, 0xA9])#A9
    command = serial.to_bytes(
        [0x09, 0x10, 0x03, 0xE8, 0x00, 0x03, 0x06, 0x09, 0x00, 0x00, postion, velocity, force, crc[0], crc[1]])
    
    ser = serial.Serial(port=COM, baudrate=115200, timeout=1,
                        parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS)
    
    ser.write(activation_request)
    time.sleep(sleep_time)
    ser.write(command)
    # data_raw = ser.readline()

def gripActivate(COM,sleep_time):
    ser = serial.Serial(port=COM, baudrate=115200, timeout=1,
                        parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS)
    ser.write(serial.to_bytes([0x09,0x10,0x03,0xE8,0x00,0x03,0x06,0x00,0x00,0x00,0x00,0x00,0x00,0x73,0x30]))
    time.sleep(sleep_time)
