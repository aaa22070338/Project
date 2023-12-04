import socket
import time
from typing import List, Optional
from robot.helpers import *
from functools import wraps


def check_connection(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.arm_connection is None:
            print("請先連接上手臂網路")
            return self
        result = func(self, *args, **kwargs)
        return result
    return wrapper

def print_update_position(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        current_position = receive(self.arm_connection)
        print("夾爪已移動至: ", current_position)
        self.position = current_position
        time.sleep(self.arm_sleep_time)
        return result
    return wrapper


class robotic_arm:
    def __init__(self, gripper_port: str, arm_connection: socket.socket | None = None) -> None:
        self.arm_connection = arm_connection
        self.gripper_port = gripper_port
        self.origin = [360,-25 , 500, -180, 0, -60]
        self.position = None
        self.arm_sleep_time = 0.05
        self.gripper_sleep_time = 1

    def set_arm_sleep_time(self, sleep_time):
        self.arm_sleep_time = sleep_time
        return self

    def set_girpper_sleep_time(self, sleep_time):
        self.gripper_sleep_time = sleep_time
        return self
    
    @check_connection
    @print_update_position
    def move_to(self, *args: List[Optional[float]], **kwargs: Optional[float]) -> 'robotic_arm':
        """
        Move the robotic arm to the specified position.

        Parameters:
        - If using a list for `position`:
            args (List[Optional[float]]): A list containing x, y, z, rx, ry, rz values.
        - If using keyword arguments for individual values:
            x (Optional[float]): X-coordinate.
            y (Optional[float]): Y-coordinate.
            z (Optional[float]): Z-coordinate.
            rx (Optional[float]): Rotation around the x-axis.
            ry (Optional[float]): Rotation around the y-axis.
            rz (Optional[float]): Rotation around the z-axis.

        Returns:
        robotic_arm: The robotic arm object.

        Example usage:
        ```
        robot.move_to([500, 200, 600, -180, 0, -60])
        # or
        robot.move_to(x=500, y=200, z=600, rx=-180, ry=0, rz=-60)
        ```
        """
        position = list(args[0]) if args else [
            kwargs.get('x', self.position[0]),
            kwargs.get('y', self.position[1]),
            kwargs.get('z', self.position[2]),
            kwargs.get('rx', self.position[3]),
            kwargs.get('ry', self.position[4]),
            kwargs.get('rz', self.position[5])
        ]
        robot_move(position, self.arm_connection)
        return self

    @check_connection
    def set_origin(self, position):
        self.origin = position
        print("設定原點為: ", self.origin)
        return self

    @check_connection
    @print_update_position
    def move_to_origin(self):
        robot_move(self.origin, self.arm_connection)
        return self

    @check_connection
    @print_update_position
    def stay(self):
        robot_move(self.position, self.arm_connection)
        return self

    @check_connection
    def terminate(self):
        robot_move([0, 0, 0, 0, 0, 0], self.arm_connection)
        return self

    def grip_activate(self):
        gripActivate(self.gripper_port, self.gripper_sleep_time)
        return self

    def grip_move(self, pos, velocity, force):
        gripMove(pos, velocity, force, self.gripper_port, self.gripper_sleep_time)
        return self

    def grip_complete_open(self):
        gripMove(0, 255, 255, self.gripper_port, self.gripper_sleep_time)
        return self

    def grip_complete_close(self):
        gripMove(255, 255, 255, self.gripper_port, self.gripper_sleep_time)
        return self


if __name__ == "__main__":
    pass
