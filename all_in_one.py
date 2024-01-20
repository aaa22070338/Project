from ultralytics import YOLO
import robot.robot as bot
import socket
import pickle
import cv2
import numpy as np
import catcher_ex as CT
import cube_detector.cube_detector as CD
import catch_save_throw
import time
import serial

#顏色順序的要求
color_input = "green"
#['green' ,'yellow','red','purple','blue']
color_list = ['green' ,'yellow','red','purple','blue']
print(color_list)
index = 0
#連線
TCP_IP = "192.168.0.1"  #  Robot IP address. Start the TCP server from the robot before starting this code
TCP_PORT = 3000  #  Robot Port
BUFFER_SIZE = 1024  #  Buffer size of the channel, probably 1024 or 4096
# gripper_port = '/dev/ttyUSB2'  # gripper USB port to linux
gripper_port = "COM15"  # gripper USB port to windows
global c
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  #  Initialize the communication the robot through TCP as a client, the robot is the server.
#  Connect the ethernet cable to the robot electric box first
c.connect((TCP_IP, TCP_PORT))
arm = bot.robotic_arm(gripper_port,c)
arm.set_arm_sleep_time(0.05)
arm.set_girpper_sleep_time(1)
arm.set_offset(3,1.5,-93)
arm.move_to_origin()
origin_x, origin_y,_,_,_,_ = arm.position
arm.move_to(rz=0)
arm.grip_complete_open()


model_part = YOLO("./cube_surface.pt")
model_region = YOLO("./cube.pt")
model = YOLO("./cube.pt")
surface_model = YOLO('./cube_surface.pt')
model_color = YOLO('./cube_color.pt')

with open('./hand_matrix/calibration.pkl', 'rb') as file:
    camera_matrix, dist_coeff = pickle.load(file)
with open("./fixedCam_matrix/MultiMatrix_fixed_640_480.npz","rb") as file:
    mtx = np.load(file)["camMatrix"]
    dist = np.load(file)["distCoef"]

CT = CT.block_detect(model_part, model_region)
detector = CD.CubeDetector(model, surface_model , model_color)
save_coor =catch_save_throw.CatchAndSave()


def draw_axis(img, corners, image_points):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(np.int16(image_points[0].ravel())), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(np.int16(image_points[1].ravel())), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(np.int16(image_points[2].ravel())), (0,0,255), 5)
    return img
def draw_xy_lines(img, image_points):
    img = cv2.line(img, tuple(np.int16(image_points[0].ravel())), tuple(np.int16(image_points[1].ravel())), (0,0,0), 5)
    img = cv2.line(img, tuple(np.int16(image_points[1].ravel())), tuple(np.int16(image_points[2].ravel())), (0,0,0), 5)
    return img
def draw_z_lines(img, image_points):
    img = cv2.line(img, tuple(np.int16(image_points[1].ravel())), tuple(np.int16(image_points[3].ravel())), (128,128,0), 5)
    return img

object_points = np.array( #中心
    [
        [-25, -25, 0],  # 1
        [-25, 25, 0],  # 2
        [25, 25, 0],  # 3
        [25, -25, 0],  # 4
    ],
    dtype=np.float32,
)
#環境相機確認座標(取10次的平均值)


fixed_to_table = np.float32(
    [[0, 1, 0, -650],
     [1, 0, 0, -250],
     [0, 0, -1, 860],
     [0, 0, 0, 1]])
_, table_rmtx, table_tvec, _, _, _, _  = cv2.decomposeProjectionMatrix(fixed_to_table[0:3,0:])

table_rvec, _ = cv2.Rodrigues(table_rmtx)
table_tvec = fixed_to_table[0:3,3]
print(table_tvec)   

# color = "yellow"
while index < len(color_list):
    color = color_list[index]
    save_coor.reset()
    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

    while True:
        cap_success, frame = cap.read()
        if cap_success:
            height = 480
            scale = height / frame.shape[0]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            #  將圖片丟入偵測器進行偵測
            result_img = detector.detect(frame, index=None, color=color)
            
            # 讀取抓到的角點座標
            for color_name in list(detector.cube_contour_outer.keys()):
                cube_imagePoints = detector.get_cube_sequence_imagePoints(color_name)
                if cube_imagePoints is None:
                    cv2.imshow("result", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue
                # 依順序定出對應真實世界對應座標
                cube_objectPoints = np.array(
                    [(0, 0, 0), (0, 50, 0), (50, 50, 0),
                    (50, 0, 0), (50, 0, -50), (0, 0, -50)]
                )
                # 如果抓到六個點採用EPNP算法(較準確穩定)，四個點則是一般算法
                print(type(cube_imagePoints),'\n',f"{cube_imagePoints}")
                if cube_imagePoints.shape[0] == 6:
                    PnP_success, cube_rvec, cube_tvec = cv2.solvePnP(
                        cube_objectPoints.astype(float), cube_imagePoints, mtx, dist, flags=cv2.SOLVEPNP_EPNP)
                elif cube_imagePoints.shape[0] == 4:
                    PnP_success, cube_rvec, cube_tvec = cv2.solvePnP(
                        cube_objectPoints[:4].astype(float), cube_imagePoints, mtx, dist)
                #  若成功計算出來則計算出該方塊相對於桌面座標系的座標位置
                if PnP_success:
                    R1, _ = cv2.Rodrigues(cube_rvec)  # 創建方塊到相機座標系的旋轉矩陣
                    R2, _ = cv2.Rodrigues(table_rvec)  # 創建桌面到方塊座標系的旋轉矩陣
                    T1 = np.eye(4)  # 創建4*4單位矩陣
                    T2 = np.eye(4)
                    T1[:3, :3], T1[:3, 3] = R1, cube_tvec.T  #  方塊座標系旋轉後平移到相機座標系 轉換矩陣
                    T2[:3, :3], T2[:3, 3] = R2, table_tvec.T  # 桌面座標系旋轉後平移到相機座標系 轉換矩陣
                    T2 = np.linalg.inv(T2)  # 取逆矩陣，變成 相譏座標系到桌面座標系 轉換矩陣
                    # fixed_to_table = np.linalg.inv(fixed_to_table)  # 取逆矩陣，變成 相譏座標系到桌面座標系 轉換矩陣
                    transform_matrix = T2 @ T1  # 結合二者，得到方塊座標系到桌面座標系的轉換矩陣
                    transformed_point = (transform_matrix @ np.array([[25, 25, -25, 1]]).T)  # 傳入方塊正中央座標，即可得到方塊在桌面座標位置了
                    # 输出结果
                    print("物體在桌面座標系下的座標:\n", transformed_point)
                # xyz座標位置線繪製
                x, y, z = transformed_point[0:3].flatten()

                xyz_line_points = np.float32([[0, y, 0], [x, y, 0], [x, 0, 0], [x, y, z]]).reshape(-1, 3)
                xyz_line_points_img, _ = cv2.projectPoints(xyz_line_points, table_rvec, table_tvec, mtx, dist)
                result_img = draw_xy_lines(result_img, xyz_line_points_img)
                result_img = draw_z_lines(result_img, xyz_line_points_img)
                print(x, y)        # xyz座標軸繪製
                axis = np.float32([[25, 0, 0], [0, 25, 0], [0, 0, 25]]).reshape(-1, 3)
                axis_cube_img_points, _ = cv2.projectPoints(axis, cube_rvec, cube_tvec, mtx, dist)
                # axis_table_img_points, _ = cv2.projectPoints(axis, table_rvec, table_tvec, mtx, dist)
                result_img = draw_axis(result_img, cube_imagePoints.astype(int), axis_cube_img_points)
                # result_img = draw_axis(result_img, table_imagePoints.astype(int), axis_table_img_points)

            # 取得方塊輪廓計算方塊重心位置
            
            
                print(color_name)
                contour_outer = detector.get_cube_contour_outer(color_name)
                radius = int(0.08 * cv2.arcLength(contour_outer, True))
                print(radius)
                M = cv2.moments(contour_outer)
                if M["m00"] != 0:
                    centroid_X = int(M["m10"] / M["m00"])  # 算形心x
                    centroid_Y = int(M["m01"] / M["m00"])  # 算形心y
                # 繪製重心座標點
                result_img = cv2.circle(result_img, (centroid_X, centroid_Y), radius, (255, 0, 255), 2)

                # 繪製PNP計算所得座標點
                predict_cube_center_imagePoint = cv2.projectPoints(np.float32([x, y, z]), table_rvec, table_tvec, mtx, dist)
                predict_cube_center_imagePoint = predict_cube_center_imagePoint[0].astype(np.int16)

                cv2.putText(result_img, f"{x=:.1f}, {y=:.1f}, {z=:.1f}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2,)

                # 推算是否位置估計錯誤
                if (np.linalg.norm(np.array((centroid_X, centroid_Y)) - predict_cube_center_imagePoint) > radius):
                    cv2.putText(result_img, f"Error", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4,)
                    result_img = cv2.circle(result_img, predict_cube_center_imagePoint.ravel(), 5, (0, 0, 255), -1)
                else:
                    cv2.putText(result_img, f"Ok", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4,)
                    result_img = cv2.circle(result_img, predict_cube_center_imagePoint.ravel(), 5, (0, 255, 0), -1)
                    print(f"{type(color_name)=}")
                    x,y,rz = transformed_point.flatten()[:3]
                    # print(f"{x=}")
                    # print(f"{y=}")
                    # print(f"{z=}")
                    save_coor.catch_save(color_name,[x],[y],[rz],9)

            cv2.imshow("result", result_img)
            Key = cv2.waitKey(1)
            if Key == 27:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if save_coor.completed_Save == True:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    #移動
    file_path = "input_file_a.json"
    environment_coor = save_coor.coordinates(file_path)
    print(environment_coor)    
    arm.move_to(y=350)
    arm.move_to(x=environment_coor[0],y=environment_coor[1]-80)
    arm.move_to(z=350)
    save_coor.reset()
    #手臂相機確認座標
    lens = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
            ret, img = lens.read()
            if not ret:
                break
            vertical_offset = 0
            
            for image_points in CT.detect_parts(img):
                for color_name, rgb in CT.get_color_text(img):
                    if not color_name == color :
                        continue
                    image_points = np.float32(image_points)
                    retval, rvec, tvec = cv2.solvePnP(object_points[:4], image_points, camera_matrix, dist_coeff)
                    if not retval:
                        break
                    x = tvec[0]
                    y = tvec[1]
                    z = tvec[2]
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    # 使用旋转矩阵计算欧拉角（roll-pitch-yaw 顺序）
                    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    pitch = np.arctan2(-rotation_matrix[2, 0],np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2),)
                    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                    # 将弧度转换为度数
                    rz = np.degrees(yaw)
                    ry = np.degrees(pitch)
                    rx = np.degrees(roll)
                    rz = np.array([rz])
                    text_loc_tvec = (5, 15 + vertical_offset)
                    text_loc_rvec = (5, 32 + vertical_offset)
                    text_loc_check = (400, 15 + vertical_offset)

                    save_coor.catch_save(color_name, [float(x)], [float(y)], [float(rz)], 9)
                    xyz_str = [f"{c}: {v[0]:.2f}" for c, v in zip("xyz", [x, y, z])]
                    cv2.putText(img, f"{color_name} {', '.join(xyz_str)}", text_loc_tvec, cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
                    # cv2.putText(img, f"Rotate Z: {rz:.1f},   Rotate Y: {ry:.1f},   Rotate X: {rz:.1f}", text_loc_rvec, cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
                    if -10 < x < 10 and -10 < y < 10:
                        cv2.putText(img, "OK", text_loc_check, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(img, "Moving", text_loc_check, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) ,2)
                    cv2.circle(img, (int(x[0]), int(y[0])), 5, (0, 0, 255), -1)
            cv2.imshow("test", img)
            cv2.waitKey(1)

            if save_coor.completed_Save == True:
                break
    lens.release()
    cv2.destroyAllWindows()

    #-------移動---------
    environment_coor = save_coor.coordinates(file_path)
    print(environment_coor)    

    if environment_coor[2] >= 45 :
        arm.cam_move_to(x=environment_coor[0],y=environment_coor[1],alpha=environment_coor[2] - 90)    
    else:    
        arm.cam_move_to(x=environment_coor[0],y=environment_coor[1],alpha=environment_coor[2])    
    arm.grip_move_to(x=-4, y=90)
    arm.move_to(z=240)
    arm.grip_complete_close()
    arm.move_to(z=260 + ((index+1)*50))
    arm.move_to(x= origin_x)
    arm.move_to(y= origin_y,rz =0)
    arm.move_to(z=230 + (index*50))
    arm.grip_complete_open()#放開
    arm.move_to_origin()

    index += 1


arm.terminate()
time.sleep(3)


