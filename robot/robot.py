import socket
import time
from robot.helpers import *


class robotic_arm:
    def __init__(self, target_arm: socket.socket, target_gripper: str) -> None:
        self.target_arm = target_arm
        self.target_gripper = target_gripper
        self.origin = [500, 200, 600, -180, 0, -60]
        self.position = None

    def move_to(self, position):
        robot_move(position, self.target_arm)
        current_position = receive(self.target_arm)
        print("夾爪已移動至: ", current_position)
        self.position = current_position
        time.sleep(1)
        return self

    def setup_origin(self, position):
        self.origin = position
        print("設定原點為: ", self.origin)
        return self

    def move_to_origin(self):
        robot_move(self.origin, self.target_arm)
        msg = receive(self.target_arm)
        print("夾爪已移動至: ", msg)
        time.sleep(1)
        return self

    def stop(self):
        robot_move([0,0,0,0,0,0],self.target_arm)
        msg = receive(self.target_arm)
        print("夾爪已移動至: ", msg)
        time.sleep(1)
        return self

    def grip_move(self, pos, velocity, force):
        gripMove(pos,velocity,force,self.target_gripper)
        return self

    def grip_complete_open(self):
        gripMove(0,255,255,self.target_gripper)
        return self

    def grip_complete_close(self):
        gripMove(255,255,255,self.target_gripper)
        return self
