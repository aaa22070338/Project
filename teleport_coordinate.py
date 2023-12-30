import numpy as np
import cv2
from ultralytics import YOLO
import cube_detector.cube_detector as CD
import pickle
import SaveSystem as SS

SS.SaveSystem.reset()

model = YOLO("./yolov8n-seg-custom.pt")
surface_model = YOLO('./cube_surface_seg2.pt')

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

detector = CD.CubeDetector(model, surface_model)
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
# with open("test_calibration.pkl", "rb") as file:
#     mtx, dist = pickle.load(file)
with open("./fixedCam_matrix/MultiMatrix_fixed_640_480.npz","rb") as file:
    mtx = np.load(file)["camMatrix"]
    dist = np.load(file)["distCoef"]
# with open('test_table_points.pkl','rb') as file:
    # table_objectPoints,table_imagePoints = pickle.load(file)
# with open('test_table_solvePnP.pkl','rb') as file:
#     table_rvec,table_tvec = pickle.load(file)
fixed_to_table = np.float32(
    [[0, 1, 0, -650],
     [1, 0, 0, -250],
     [0, 0, -1, 860],
     [0, 0, 0, 1]]
)
_, table_rmtx, table_tvec, _, _, _, _  = cv2.decomposeProjectionMatrix(fixed_to_table[0:3,0:])

table_rvec, _ = cv2.Rodrigues(table_rmtx)
table_tvec = fixed_to_table[0:3,3]
print(table_tvec)   

# color = "yellow"
while True:
    cap_success, frame = cap.read()
    if cap_success:
        height = 480
        scale = height / frame.shape[0]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        #  將圖片丟入偵測器進行偵測
        result_img = detector.detect(frame, index=None, color=None)
        
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
                x,y,z = transformed_point[0:3]
                print(f"{x=}")
                print(f"{y=}")
                print(f"{z=}")
                SS.SaveSystem.save_coordinate(color_name,x, y, z)

        cv2.imshow("result", result_img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            break
    else:
        break
cv2.destroyAllWindows()
