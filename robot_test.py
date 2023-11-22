import robot.robot as bot
import socket

TCP_IP = "192.168.0.1"  #  Robot IP address. Start the TCP server from the robot before starting this code
TCP_PORT = 3000  #  Robot Port
BUFFER_SIZE = 1024  #  Buffer size of the channel, probably 1024 or 4096

# gripper_port = '/dev/ttyUSB1'  # gripper USB port to linux
gripper_port = "COM8"  # gripper USB port to windows

global c
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  #  Initialize the communication the robot through TCP as a client, the robot is the server.
#  Connect the ethernet cable to the robot electric box first
c.connect((TCP_IP, TCP_PORT))

arm = bot.robotic_arm(gripper_port,c)
arm.move_to_origin()
arm.grip_activate()
arm.grip_move(128, 128, 128)
arm.grip_complete_open()
arm.terminate()