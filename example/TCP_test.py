import re
import socket

def robot_move(x, y, z, rx, ry, rz, c):
    mesg = f'{x},{y},{z},{rx},{ry},{rz}'
    c.send(bytes(mesg, "utf-8"))
def receive(c):
    data = c.recv(1024).decode()
    data = re.sub(',', ' ', data)
    pos = data.split()
    pos = list(map(float, pos))  # [x, y, z, rx - psi, ry - theta, rz - phi]
    return pos

TCP_IP = '192.168.0.1' #  Robot IP address. Start the TCP server from the robot before starting this code
TCP_PORT = 3000  #  Robot Port
BUFFER_SIZE = 1024  #  Buffer size of the channel, probably 1024 or 4096
global c
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  #  Initialize the communication the robot through TCP as a client, the robot is the server. 
                                                       #  Connect the ethernet cable to the robot electric box first
c.connect((TCP_IP, TCP_PORT)) 

robot_move(0, 0, 1071, 0, 0, 0, c)  #  傳送座標
rPos = receive(c) #接收座標
print(rPos)