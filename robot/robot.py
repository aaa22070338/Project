import socket
import time
from typing import List, Optional

import numpy as np
from robot.helpers import *
from functools import wraps
import copy



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
        current_position = self.sub_offset(current_position)
        print("夾爪已移動至: ", f"{current_position[0]:.2f}, {current_position[1]:.2f}, {current_position[2]:.2f}, {current_position[3]:.2f}, {current_position[4]:.2f}, {current_position[5]:.2f}")
        self._position = current_position
        time.sleep(self.arm_sleep_time)
        return result
    return wrapper


class robotic_arm:
    def __init__(self, gripper_port: str, arm_connection: socket.socket | None = None) -> None:
        self.arm_connection = arm_connection
        self.gripper_port = gripper_port
        self.ser = serial.Serial(self.gripper_port,9600,timeout=1)
        time.sleep(2)
        self.origin = [360-26,-25-13+4 , 500, -180, 0, 0]
        self.position = None
        self.arm_sleep_time = 0.05
        self.gripper_sleep_time = 1
        self.rx_offset = 0
        self.ry_offset = 0
        self.rz_offset = 0
        self.C2G_transfer_matrix = np.array([
            [-1, 0, 7.6787],
            [0, 1, 96.4183],
            [0, 0, 1]
            # [-1, 0, 1.18514],
            # [0, 1, 32.6082],
            # [0, 0, 1]
        ])

    # @check_connection
    @property
    def position(self):
        if self._position is None:
            robot_move([0, 0, 0, 0, 0, 1], self.arm_connection)

            current_position = receive(self.arm_connection)
            current_position = self.sub_offset(current_position)
            print("夾爪已移動至: ", f"{current_position[0]:.2f}, {current_position[1]:.2f}, {current_position[2]:.2f}, {current_position[3]:.2f}, {current_position[4]:.2f}, {current_position[5]:.2f}")
            self._position = current_position
            time.sleep(self.arm_sleep_time)
            return self._position
        else:
            return self._position

    @position.setter
    def position(self, position):
        self._position = position

    def set_arm_sleep_time(self, sleep_time):
        self.arm_sleep_time = sleep_time
        return self

    def set_girpper_sleep_time(self, sleep_time):
        self.gripper_sleep_time = sleep_time
        return self

    def set_C2G_transfer_matrix(self, matrix=None, x=None, y=None):
        """
        Sets the Camera to Gripper transfer matrix.

        Args:
            matrix (np.ndarray, optional): The transfer matrix to set. Defaults to None.
            x (float, optional): The x value to use to create the transfer matrix. Defaults to None.
            y (float, optional): The y value to use to create the transfer matrix. Defaults to None.

        Returns:
            self: The updated object.
        ```
        Matix form of C2G_transfer_matrix:
        [-1, 0, x],
        [ 0, 1, y],
        [ 0, 0, 1]
        ``` 
        """

        if matrix is not None:
            self.C2G_transfer_matrix = matrix
        elif x is not None and y is not None:
            self.C2G_transfer_matrix = np.array([
                [-1, 0, x],
                [0, 1, y],
                [0, 0, 1]
            ])
        else:
            raise ValueError("Please provide either a matrix or x and y values.")
        return self

    def set_origin(self, position):
        self.origin = position
        print("設定原點為: ", self.origin)
        return self
    def set_offset(self, rx_offset, ry_offset, rz_offset):
        self.rx_offset = rx_offset
        self.ry_offset = ry_offset
        self.rz_offset = rz_offset
        print(f"設定偏移量為: {rx_offset=}, {ry_offset=}, {rz_offset=}")
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
        position = self.__add_offset(position)
        robot_move(position, self.arm_connection)
        return self

    @check_connection
    @print_update_position
    # 夾抓移動，以增量方式
    def move_by(self, *args: List[Optional[float]], **kwargs: Optional[float]) -> 'robotic_arm':
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
        robot.move_by([20, 30, 60, 0, 0, 0])
        # or with keyword arguments
        robot.move_by(x=20, y=30, z=60, rx=0, ry=0, rz=0)
        # or with partial keyword arguments
        robot.move_by(rx=0, ry=0, rz=38)
        ```
        """
        position = list(args[0]) if args else [
            self.position[0] + kwargs.get('x', 0),
            self.position[1] + kwargs.get('y', 0),
            self.position[2] + kwargs.get('z', 0),
            self.position[3] + kwargs.get('rx', 0),
            self.position[4] + kwargs.get('ry', 0),
            self.position[5] + kwargs.get('rz', 0)
        ]
        position = self.__add_offset(position)
        robot_move(position, self.arm_connection)
        return self

    @check_connection
    @print_update_position
    def move_to_origin(self, move_z:str | None = 500):
        position = self.__add_offset(copy.deepcopy(self.origin))
        if move_z:
            position[2] = move_z
        robot_move(position, self.arm_connection)
        return self

    @check_connection
    @print_update_position
    def stay(self):
        position = self.__add_offset(self.position)
        robot_move(position, self.arm_connection)
        return self

    @check_connection
    @print_update_position
    def cam_move_to(self, x, y, alpha=None):
        current_position = self.position
        rz = np.radians(current_position[5])
        G2A_transfer_matrix = np.array([  # gripper to arm
            [np.cos(rz), -np.sin(rz), current_position[0]],
            [np.sin(rz), np.cos(rz), current_position[1]],
            [0, 0, 1]
        ])
        # 將相譏、目標點位置轉到手臂座標系
        cam_position_atGrip = self.C2G_transfer_matrix @ np.array([0, 0, 1]).T
        cam_position = G2A_transfer_matrix @ cam_position_atGrip 

        target_position_atGrip =  self.C2G_transfer_matrix @ np.array([x, y, 1]).T
        target_position = G2A_transfer_matrix @ target_position_atGrip
        # # 計算兩位置機械座標差值
        delta_x = target_position[0] - cam_position[0]
        delta_y = target_position[1] - cam_position[1]
        # # 移動該差值
        position = current_position
        position[0] += delta_x
        position[1] += delta_y
        # 若有給角度，則將相機座標轉到該角度
        if alpha is not None:
            phi = rz-np.radians(alpha)
            # 將相譏、目標點位置轉到手臂座標系
            cam_position_atGrip = self.C2G_transfer_matrix @ np.array([0, 0, 1]).T
            # 計算要移動至的機械座標
            position[0] = target_position[0]-np.cos(phi)*cam_position_atGrip[0]+np.sin(phi)*cam_position_atGrip[1]
            position[1] = target_position[1]-np.sin(phi)*cam_position_atGrip[0]-np.cos(phi)*cam_position_atGrip[1]
            position[5] = np.degrees(phi)

        position = self.__add_offset(position)
        robot_move(position, self.arm_connection)
        return self
 
    @check_connection
    @print_update_position
    def cam_rotate_to(self, alpha):
        current_position = self.position
        rz = np.radians(current_position[5])
        phi = rz-np.radians(alpha)
        
        G2A_transfer_matrix = np.array([  # gripper to arm
            [np.cos(rz), -np.sin(rz), current_position[0]],
            [np.sin(rz), np.cos(rz), current_position[1]],
            [0, 0, 1]
        ])
        # 將相譏、目標點位置轉到夾抓、手臂座標系
        cam_position_atGrip = self.C2G_transfer_matrix @ np.array([0, 0, 1]).T
        cam_position = G2A_transfer_matrix @ cam_position_atGrip
        # 計算要移動至的機械座標
        position = current_position
        position[0] = cam_position[0]-np.cos(phi)*cam_position_atGrip[0]+np.sin(phi)*cam_position_atGrip[1]
        position[1] = cam_position[1]-np.sin(phi)*cam_position_atGrip[0]-np.cos(phi)*cam_position_atGrip[1]
        position[5] = np.degrees(phi)
        # 移動
        position = self.__add_offset(position)
        robot_move(position, self.arm_connection)
        return self

    @check_connection
    @print_update_position
    def grip_move_to(self, x, y, z: None = None):
        current_position = self.position
        rz = np.radians(current_position[5])
        G2A_transfer_matrix = np.array([  # gripper to arm
            [np.cos(rz), -np.sin(rz), current_position[0]],
            [np.sin(rz), np.cos(rz), current_position[1]],
            [0, 0, 1]
        ])
        grip_position = G2A_transfer_matrix @ np.array([x, y, 1]).T
        position = current_position
        position[0] = grip_position[0]
        position[1] = grip_position[1]
        position = self.__add_offset(position)
        if z:
            position[2] = z
        robot_move(position, self.arm_connection)
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
        # gripMove(0, 255, 255, self.gripper_port, self.gripper_sleep_time)
        val = self.ser.write("0".encode("utf-8"))
        time.sleep(self.gripper_sleep_time)
        return self

    def grip_complete_close(self):
        # gripMove(255, 255, 255, self.gripper_port, self.gripper_sleep_time)
        val = self.ser.write("1".encode("utf-8"))
        time.sleep(self.gripper_sleep_time)
        return self

    def __add_offset(self, position: list):
        position[3] += self.rx_offset
        position[4] += self.ry_offset
        position[5] += self.rz_offset
        return position

    def sub_offset(self, position: list):
        position[3] -= self.rx_offset
        position[4] -= self.ry_offset
        position[5] -= self.rz_offset
        return position

if __name__ == "__main__":
    pass
