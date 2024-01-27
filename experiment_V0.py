from ultralytics import YOLO
import robot.robot as bot
import socket
import pickle
import cv2
import numpy as np
import catcher_ex as CT
import cube_detector.cube_detector as CD
import SaveSystem_by_environment
import SaveSystem_by_grip
import time
#-------------連線-------------
TCP_IP = "192.168.0.1"  #  Robot IP address. Start the TCP server from the robot before starting this code
TCP_PORT = 3000  #  Robot Port
BUFFER_SIZE = 1024  #  Buffer size of the channel, probably 1024 or 4096

# gripper_port = '/dev/ttyUSB2'  # gripper USB port to linux
gripper_port = "COM15"  # gripper USB port to windows

global c
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  #  Initialize the communication the robot through TCP as a client, the robot is the server.
#  Connect the ethernet cable to the robot electric box first
c.connect((TCP_IP, TCP_PORT))

arm = bot.robotic_arm(gripper_port ,c)
arm.set_arm_sleep_time(0.05)
arm.set_girpper_sleep_time(1)
arm.set_offset(3,2.5,-93)
# arm.set_origin([360-21,-25-13 , 500, -180, 0, 0])
arm.move_to_origin()
arm.move_to(rz=0)
arm.grip_complete_open()

cube_model = YOLO("./cube.pt")
surface_model = YOLO('./cube_surface.pt')

model_color_grip = YOLO("./grip_cube_color.pt")
model_color_emvironment = YOLO('./cube_color.pt')

with open('./hand_matrix/calibration.pkl', 'rb') as file:
    camera_matrix, dist_coeff = pickle.load(file)
with open("./fixedCam_matrix/MultiMatrix_fixed_640_480.npz","rb") as file:
    mtx = np.load(file)["camMatrix"]
    dist = np.load(file)["distCoef"]

series : list[CD.ColorType] = ["yellow", 'purple']#---------------------------------------------------------------------------------------------------------------------------顏色輸入

CT = CT.block_detect(surface_model, cube_model, model_color_grip)
detector = CD.CubeDetector(cube_model, surface_model, model_color_emvironment) 
Save_2_environ = SaveSystem_by_environment.save_system(series)
Save_2_grip = SaveSystem_by_grip.save_system()

Save_2_environ.reset()

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
    ], dtype=np.float32)
#-------------------抓環境座標----------------------
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)

fixed_to_table = np.float32(
    [[0, 1, 0, -600],
     [1, 0, 0, -260],
     [0, 0, -1, 940],
     [0, 0, 0, 1]])
_, table_rmtx, table_tvec, _, _, _, _  = cv2.decomposeProjectionMatrix(fixed_to_table[0:3,0:])

table_rvec, _ = cv2.Rodrigues(table_rmtx)
table_tvec = fixed_to_table[0:3,3]
# print(table_tvec)   

while True:
    cap_success, frame = cap.read()
    if cap_success:
        height = 480
        scale = height / frame.shape[0]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        #  將圖片丟入偵測器進行偵測
        result_img = detector.detect(frame, index=None, color=None,show_process_img=False)
        
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
                # print(f"{type(color_name)=}")
                x,y,z = transformed_point[0:3]
                # print(f"{x=}")
                # print(f"{y=}")
                # print(f"{z=}")
                Save_2_environ.save_coordinate_by_color(color_name, x, y, 8)

        cv2.imshow("result", result_img)
        Key = cv2.waitKey(1)
        if Key == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if Save_2_environ.completed_Save == True:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
isFirst = True
#-------------------取出環境座標並移動-------------------

environment_coor = Save_2_environ.get_coordinates_by_color(series)
print(environment_coor)
for i in range(len(environment_coor)):
    pile_z_axis = 230 + (i*50)

    x = environment_coor[i][0]
    y = environment_coor[i][1]
    arm.move_to(y=400)
    arm.move_to(x=x, y=y-80, z=350)
    Save_2_grip.reset()
    #------------------夾爪抓偏移座標並移動---------------------
    lens = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        ret, img = lens.read()
        if not ret:
            break
        vertical_offset = 0
        for image_points, color_name, rgb in CT.detect_surface(img):
            #for color_name, rgb in CT.get_color_text(img):
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
            pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            # 将弧度转换为度数
            rz = np.degrees(yaw)
            ry = np.degrees(pitch)
            rx = np.degrees(roll)
            RotationZ = np.array([rz])
            text_loc_tvec = (5, 15 + vertical_offset)
            text_loc_rvec = (5, 32 + vertical_offset)
            text_loc_check = (400, 15 + vertical_offset)

            Save_2_grip.save_coordinate_by_color(series[i], x, y, RotationZ, 15)
            xyz_str = [f"{c}: {v[0]:.2f}" for c, v in zip("xyz", [x, y, z])]
            cv2.putText(img, f"{color_name} {', '.join(xyz_str)}", text_loc_tvec, cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
            cv2.putText(img, f"Rotate Z: {rz:.1f},   Rotate Y: {ry:.1f},   Rotate X: {rx:.1f}", text_loc_rvec, cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
            if -10 < x < 10 and -10 < y < 10:
                cv2.putText(img, "OK", text_loc_check, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Moving", text_loc_check, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) ,2)
            vertical_offset += 34
        cv2.imshow("test", img)
        cv2.waitKey(1)
        if Save_2_grip.completed_Save == True:
            break
    lens.release()
    cv2.destroyAllWindows()

    grip_coor = Save_2_grip.get_coordinates_by_color(series[i])
    print(grip_coor)
    for j in range(len(grip_coor)):
        offset_x = grip_coor[j][0]
        offset_y = grip_coor[j][1] 
        offset_Rz = grip_coor[j][2]
        if offset_Rz >= 45:
            offset_Rz = offset_Rz -90
        print(offset_x, offset_y, offset_Rz)
        arm.cam_move_to(x=offset_x, y=offset_y, alpha=offset_Rz)
    #------------------向下並抓起移回原點--------------------
        arm.grip_move_to(x=-5, y=95, z=240)
        arm.grip_complete_close()#夾起
        if isFirst:
            arm.move_to(z=pile_z_axis + 80)#往上移
            arm.move_to(x=350, y=400)
            arm.move_to_origin(move_z=pile_z_axis + 80)#移動到放置位置
            arm.move_to(z=pile_z_axis-5)#向下推
            isFirst = False
        else:
            arm.move_to(z=pile_z_axis+25)#往上移
            arm.move_to(x=350, y=400)
            arm.move_to_origin(move_z=pile_z_axis+15)#移動到放置位置
            arm.move_to(z=pile_z_axis-5)#向下推
        arm.grip_complete_open()#放開
        arm.move_to_origin(move_z=470)

arm.move_to_origin()
arm.terminate()
time.sleep(3)

