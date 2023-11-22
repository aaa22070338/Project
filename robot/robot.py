import socket
import time
from robot.helpers import *

class robotic_arm:
    def __init__(self, gripper_port: str, arm_connection: socket.socket|None=None) -> None:
        self.arm_connection = arm_connection
        self.gripper_port = gripper_port
        self.origin = [500, 200, 600, -180, 0, -60]
        self.position = None
        self.sleep_time = 1

    def move_to(self, position):
        if self.arm_connection is None:
            print("請先連接到夾爪")
            return self
        
        robot_move(position, self.arm_connection)
        current_position = receive(self.arm_connection)
        print("夾爪已移動至: ", current_position)
        self.position = current_position
        time.sleep(self.sleep_time)
        return self

    def setup_origin(self, position):
        if self.arm_connection is None:
            print("請先連接到夾爪")
            return self
        
        self.origin = position
        print("設定原點為: ", self.origin)
        return self

    def move_to_origin(self):
        if self.arm_connection is None:
            print("請先連接到夾爪")
            return self
        
        robot_move(self.origin, self.arm_connection)
        msg = receive(self.arm_connection)
        print("夾爪已移動至: ", msg)
        time.sleep(self.sleep_time)
        return self

    def terminate(self):
        if self.arm_connection is None:
            print("請先連接到夾爪")
            return self
        
        robot_move([0, 0, 0, 0, 0, 0], self.arm_connection)
        msg = receive(self.arm_connection)
        print("夾爪已移動至: ", msg)
        time.sleep(self.sleep_time)
        return self

    def grip_activate(self):
        gripActivate(self.gripper_port, self.sleep_time)
        return self

    def grip_move(self, pos, velocity, force):
        gripMove(pos, velocity, force, self.gripper_port, self.sleep_time)
        return self

    def grip_complete_open(self):
        gripMove(0, 255, 255, self.gripper_port, self.sleep_time)
        return self

    def grip_complete_close(self):
        gripMove(255, 255, 255, self.gripper_port, self.sleep_time)
        return self


if __name__ == "__main__":
    pass
    
