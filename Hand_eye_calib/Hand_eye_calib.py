'''
參考資料：http://www.lxshaw.com/tech/2023/03/28/pythonopencv%E6%89%8B%E7%9C%BC%E6%A0%87%E5%AE%9A/
'''
import numpy as np
from math import *
import cv2
#from cv2 import aruco
import pandas as pd
import socket
import time


def getarray_Diego():  # Function to get points from excel
    xl = pd.ExcelFile("Points.xlsx")
    df = xl.parse("Sheet1")
    x_matrix = []
    y_matrix = []
    z_matrix = []
    rx_matrix = []
    ry_matrix = []
    rz_matrix = []
    for X, Y, Z, Rx, Ry, Rz in zip(df["x"], df["y"], df["z"], df["rx"], df["ry"], df["rz"]):
        x_matrix.append(X)
        y_matrix.append(Y)
        z_matrix.append(Z)
        rx_matrix.append(Rx)
        ry_matrix.append(Ry)
        rz_matrix.append(Rz)
    return x_matrix, y_matrix, z_matrix, rx_matrix, ry_matrix, rz_matrix, len(x_matrix)  # #


def imReady():
    c.send(bytes("1", "utf-8"))


def robot_ready():
    x = int(c.recv(1024).decode())
    return x


TCP_IP = "192.168.0.1"  # Robot IP
TCP_PORT = 3000  # Robot Port
#data = np.load("calib_data/MultiMatrix_3D_640_480.npz")  # Load camera matrix
data = np.load("MultiMatrix_hand_640_480.npz")  # Load camera matrix
camMatrix = data["camMatrix"]
distortion = data["distCoef"]
#cap = realsense_depth.DepthCamera()  # Camera initialization
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
width_fixed = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_fixed = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"影像寬度：{width_fixed}")     #640
print(f"影像高度：{height_fixed}")    #480
param = cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)  # Aruco dictionary
detector = cv2.aruco.ArucoDetector(dictionary, param)
#square = 29.71  # Square size in mm
square = 31  # Square size in mm
markers = 24  # Marker size in mm
board = cv2.aruco.CharucoBoard((6, 9), square, markers, dictionary)  # Here you can change ChAruco dimensions
global c
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.connect((TCP_IP, TCP_PORT))
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
xm, ym, zm, rxm, rym, rzm, length = getarray_Diego()  # Get matrices from Excel
print(xm, ym, zm, rxm, rym, rzm, length)
n = 1
image_dir_path = "hand_eye_images"  ##########改
Rotation_c_aruco_arrayofarrays = np.array([])
translation_c_aruco_arrayofarrays = np.array([])
Rotation_o_t_arrayofarrays = np.array([])
translation_o_t_arrayofarrays = np.array([[]])
variable = 0
while True:
    
    if variable == 0:
        variable = robot_ready()    #接收 "1"

    if variable == 1:
        print("Getting image")
        #_, _, frame = cap.get_frame()  # You may need to change this line if you are not using an intel realsense camera
        for i in range(20):     #影時候會用上一張圖片，因此多拍一些來預防
            _, frame = cap.read()
            copyframe = frame.copy()
        grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, reject = detector.detectMarkers(grayimg)
        cv2.aruco.refineDetectedMarkers(grayimg, board, corners, ids, reject)  # Detect Aruco
        if ids is not None:
            
            charucoret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, grayimg, board)
            cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds,
                                             (0, 255, 0))
#           if key == ord("s"):  # When "s" is pressed, get rotation and translation vectors from ChAruco and robot
            ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camMatrix, distortion,
                                                             None, None)
            if rvec is not None and tvec is not None:
                cv2.drawFrameAxes(frame, camMatrix, distortion, rvec, tvec, 100)
                cv2.imshow("Camera_aruco", frame)
                Rotation_matrix, _ = cv2.Rodrigues(rvec)    #將rvec轉成旋轉矩陣
                print(f"tvec: {tvec}")
                print(f"Rotation: {np.round(Rotation_matrix, 3)}")
                #把excel的位置、角度轉成轉移矩陣====================
                psi = rxm[n-1]*np.pi/180
                theta = rym[n-1]*np.pi/180
                phi = rzm[n-1]*np.pi/180
                x = xm[n-1]
                y = ym[n-1]
                z = zm[n-1]
                Rotation_o_t = np.array([[cos(phi) * cos(theta), -sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi),
                                          sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi)],
                                         [sin(phi) * cos(theta), cos(phi) * cos(psi) + sin(phi) * sin(theta) * sin(psi),
                                          -cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi)],
                                         [-sin(theta), cos(theta) * sin(psi), cos(theta) * cos(psi)]])
                translation_o_t = np.array([[x], [y], [z]])
                translation_o_t_arrayofarrays = np.append(translation_o_t_arrayofarrays, translation_o_t)
                Rotation_o_t_arrayofarrays = np.append(Rotation_o_t_arrayofarrays, Rotation_o_t)
                print(f"Rotation from world to tool: \n{np.round(Rotation_o_t, 6)}")
                print(f"Translation {translation_o_t}")
                #到這裡==============================================
                Rotation_c_aruco_arrayofarrays = np.append(Rotation_c_aruco_arrayofarrays, Rotation_matrix)
                translation_c_aruco_arrayofarrays = np.append(translation_c_aruco_arrayofarrays, tvec)
                print(f"Array de arrays: \n{translation_o_t_arrayofarrays}")
                cv2.imwrite(f"{image_dir_path}/image{n}_{x}.png", frame)
                n = n + 1
                variable = 0
                imReady()   #送出 "1"
        key = cv2.waitKey(3)    #Refresh frame every 3 ms
        if n > length:
            break
        if key == ord("q"):
            break
cv2.destroyAllWindows()
translation_o_t_arrayofarrays = translation_o_t_arrayofarrays.reshape(length, 3)
Rotation_o_t_arrayofarrays = Rotation_o_t_arrayofarrays.reshape((length, 3, 3))
Rotation_c_aruco_arrayofarrays = Rotation_c_aruco_arrayofarrays.reshape((length, 3, 3))
translation_c_aruco_arrayofarrays = translation_c_aruco_arrayofarrays.reshape(length, 3)

print(np.round(Rotation_o_t_arrayofarrays, 5))
print(np.round(translation_o_t_arrayofarrays, 5))
print(np.round(Rotation_c_aruco_arrayofarrays, 5))
print(np.round(translation_c_aruco_arrayofarrays, 5))

R_cam2_gripper, t_cam2_gripper = cv2.calibrateHandEye(Rotation_o_t_arrayofarrays, translation_o_t_arrayofarrays,
                                                      Rotation_c_aruco_arrayofarrays, translation_c_aruco_arrayofarrays)

print(f"Rotation from camera to gripper: \n{R_cam2_gripper}")
print(f"translation from camera to gripper: \n{t_cam2_gripper}")
R_cam2_gripper2 = np.append(R_cam2_gripper, [[0, 0, 0]], axis=0)
t_cam2_gripper2 = np.append(t_cam2_gripper, [[1]], axis=0)
Homogeneous = np.append(R_cam2_gripper2, t_cam2_gripper2, axis=1)
np.savez("Hcam2grip", h_c2g=Homogeneous, r_c2g=R_cam2_gripper2, t_c2g=t_cam2_gripper2)
