import numpy as np
from math import *
import cv2
#from cv2 import aruco
import os
import openpyxl
import pandas as pd


def img_undistort(img, mtx, dist):
    """
    图像去畸变
    """
    return cv2.undistort(img, mtx, dist, None, mtx)

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
#board = aruco.CharucoBoard((6, 9), square, markers, dictionary)  # Here you can change ChAruco dimensions
board = cv2.aruco.CharucoBoard((6, 9), square, markers, dictionary)  # Here you can change ChAruco dimensions

while True:
    #_, _, frame = cap.get_frame()  # You may need to change this line if you are not using an intel realsense camera
    _, frame = cap.read()
    copyframe = frame.copy()
    # sdf = img_undistort(frame,camMatrix,distortion)
    # cv2.imshow("567",sdf)
    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, reject = detector.detectMarkers(grayimg)
    #print(ids)
    cv2.aruco.refineDetectedMarkers(grayimg, board, corners, ids, reject)  # Detect Aruco
    if ids is not None:
        # 插值Charuco角点
        charucoret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, grayimg, board)
        # 在图像上绘制检测到的Charuco角点
        cv2.aruco.drawDetectedCornersCharuco(copyframe, charucoCorners, charucoIds,
                                           (0, 255, 0))
        cv2.aruco.drawDetectedMarkers(copyframe, corners, ids)  # 绘制检测到的ArUco标记

#           if key == ord("s"):  # When "s" is pressed, get rotation and translation vectors from ChAruco and robot
        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camMatrix, distortion,
                                                            None, None)
        # print("rvec=",rvec)
        # print("tvec=",tvec)
        if rvec is not None and tvec is not None:
                cv2.drawFrameAxes(frame, camMatrix, distortion, rvec, tvec, 100)
                cv2.imshow("Camera_aruco", frame)
        
        #cv2.aruco.drawDetectedMarkers(copyframe, corners, ids)  # 绘制检测到的ArUco标记

    cv2.imshow("123",copyframe)
    key = cv2.waitKey(3)    # Refresh frame every 3 ms 
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

