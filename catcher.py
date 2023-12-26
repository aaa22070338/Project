from ultralytics import YOLO
import numpy as np
import supervision as sv
import cv2
from scipy.spatial import cKDTree
import pickle
with open('./hand_matrix/calibration.pkl', 'rb') as file:
    camera_matrix, dist_coeff = pickle.load(file)
# with open('./hand_matrix/camMatrix.npy', 'rb') as file:
#     camera_matrix = np.load(file)
# with open('./hand_matrix/distCoef.npy', 'rb') as file:
#     dist_coeff = np.load(file)

def revise_points(approx_points, epsilon):#合併點
    pile = np.vstack([point for points in approx_points 
                      for point in points])
    kdtree = cKDTree(pile)
    combine_points = [] 
    ignore_point = set()
    for i, correct in enumerate(pile):
        if i not in ignore_point:
            region_indices = kdtree.query_ball_point(correct, r=epsilon)

            ignore_point.update(region_indices)

            ave_x = sum(pile[idx][0] for idx in region_indices) / len(region_indices)
            ave_y = sum(pile[idx][1] for idx in region_indices) / len(region_indices)

            combine_points.append((ave_x, ave_y))
    
    merge_lines = []
    for point_set in approx_points:
        for i in range(len(point_set)):

            establish_points = np.array(point_set[i])

            distance = np.linalg.norm(combine_points - establish_points, axis=1)
            nearest_points = np.argmin(distance)

            point_set[i] = combine_points[nearest_points]
            merge_lines.append(point_set)
    
    return merge_lines, combine_points
  
class block_detect:
    def __init__(self, surface_model:YOLO, cube_model:YOLO) -> None:#傳入模型
            self.surface_model = surface_model
            self.cube_model = cube_model
            self.first_point = None 
            self.next_reference_point = None
            self.isFirstFrame = True

    def color_check(self, masked_img):#抓方塊顏色
        # hsv_boundary = {
        #     "white": np.array([[0, 15, 200], [75, 70, 255]], dtype=np.uint8),
        #     # "white": np.array([[75, 0, 0], [179, 100, 255]], dtype=np.uint8),
        #     "red": np.array([[122, 63, 56], [132, 255, 255]], dtype=np.uint8),
        #     "orange": np.array([[100, 40, 170], [120, 255, 255]], dtype=np.uint8),
        #     # "orange": np.array([[0, 25, 0], [20, 255, 255]], dtype=np.uint8),
        #     "yellow": np.array([[80, 90, 100], [100, 255, 255]], dtype=np.uint8),
        #     "green": np.array([[50, 150, 60], [80, 210, 200]], dtype=np.uint8),
        #     "blue": np.array([[0, 70, 25], [40, 255, 210]], dtype=np.uint8)
        # }
        
        hsv_boundary = {
            "red": np.array([[0, 138, 71], [18, 250, 250]], dtype=np.uint8),
            "green": np.array([[37, 0, 0], [76, 255, 104]], dtype=np.uint8),
            "yellow": np.array([[15, 158, 128], [109, 255, 233]], dtype=np.uint8),
            "purple": np.array([[87, 0, 81], [179, 255, 133]], dtype=np.uint8)
        }
        color_deter = {
            "red": (0, 0, 255),
            # "blue": (255, 0, 0),
            "green": (0, 128, 0),
            "yellow": (0, 255, 255),
            "purple": (160, 0, 160)
        }
        hsv = cv2.cvtColor(masked_img, cv2.COLOR_RGB2HSV)
    
        col_deter = {}
        for col_name, hsv_boun in hsv_boundary.items():
            check = cv2.inRange(hsv, hsv_boun[0], hsv_boun[1])
            pixel = cv2.countNonZero(check)
            col_deter[col_name] = pixel

        self.check_col = max(col_deter, key=col_deter.get)
        
        return color_deter[self.check_col]
    
    def detect_region(self, img, options=None):#回傳整體白色區域 or 輪廓，全部或選擇
        self.img = img
        self.predict_region = self.cube_model(self.img,verbose=False)[0]
        self.detections_region = sv.Detections.from_yolov8(self.predict_region)
        if self.detections_region.mask is None:
            return None
        #自選方塊
        if options is not None:
            region_mask = self.detections_region.mask[options].astype(np.int32)
            region_white_mask = np.where(region_mask != 0, 255, 0).astype(np.uint8)
            self.region_mask = cv2.bitwise_and(self.img, self.img, mask=region_white_mask)
            region_contours, _ = cv2.findContours(region_white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            yield self.region_mask, region_contours
        #全部方塊
        else:
            for region_mask in self.detections_region.mask:
                region_white_mask = np.where(region_mask != 0, 255, 0).astype(np.uint8)
                self.region_mask = cv2.bitwise_and(self.img, self.img, mask=region_white_mask)
                region_contours, _ = cv2.findContours(region_white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                yield self.region_mask, region_contours

    def detect_parts(self, img, options=None):#回傳部分輪廓 ,算重心
        self.img = img
        self.predict_part = self.surface_model(self.img,verbose=False)[0]
        self.detections_part = sv.Detections.from_yolov8(self.predict_part)
        
        if self.detections_part.mask is None:
            return None
        for region_mask, region_contours in self.detect_region(self.img, options):
            if region_mask is None and region_contours is None:
                return None

            planes = []
            for part_mask in self.detections_part.mask:
                part_white_mask = np.where(part_mask != 0, 255, 0).astype(np.uint8)  
                part_contours, _ = cv2.findContours(part_white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                for part_contour in part_contours:
                    M = cv2.moments(part_contour)
                    if M['m00'] and region_contours:
                        center_X = int(M['m10'] / M['m00'])
                        center_Y = int(M['m01'] / M['m00'])
                        internal = cv2.pointPolygonTest(region_contours[0], (center_X, center_Y), False)
                        if internal <= 0:
                            continue
                        planes.append(part_contour)
            
            _ , large_point = self.large_plane_process(region_mask, planes)
            if large_point is None or len(large_point) != 4:
                return None
            image_points = self.roll_image_points(large_point)
            for point in image_points:
                x, y = point
                x = int(x)
                y = int(y)
                cv2.circle(img, (x, y), 5, self.color_check(region_mask), -1)
            yield image_points
                
    def large_plane_process(self, size, planes):#最大面處理
        self.size = size
        approx_points = []
        sorted_plane = sorted(planes,key=cv2.contourArea,reverse=True)
        sorted_plane = sorted_plane[:1]
        merge_lines = None
        combine_points = None
        try:
            # sorted_plane = np.array(sorted_plane)  
            # sorted_plane = sorted_plane.reshape(-1, 2)
            epsilon_out = 0.1 * cv2.arcLength(sorted_plane[0], True)
            epsilon = 0.02 * cv2.arcLength(sorted_plane[0], True)
            approx_point = cv2.approxPolyDP(sorted_plane[0], epsilon, True)
            approx_points.append(approx_point)
            merge_lines, combine_points = revise_points(approx_points, epsilon_out)
        except IndexError:
            return None ,None
        return merge_lines, combine_points

    def get_color_text(self, img):#取方塊顏色作為文字顏色
        self.img = img
        color_deter = {
        "white": (255, 255, 255),
        "red": (0, 0, 255),
        "orange": (0, 100, 255),
        "yellow": (0, 255, 255),
        "green": (0, 128, 0),
        "blue": (255, 0, 0),
        "purple": (160, 0, 160)
        }
        
        rgb = self.color_check(self.region_mask)
        for color_name, rgb_value in color_deter.items():
            if rgb_value == rgb:
                yield f"{color_name.capitalize()}" , rgb

    def __find_reference_point(self, series):#抓前一個參考點
        if self.isFirstFrame == True:#抓第一偵參考點
            self.first_point = series[0]
            self.next_reference_point = series[0]
            self.distance = np.linalg.norm(series[0] - series[1])*0.5#抓10%的範圍
        elif self.isFirstFrame == False: #抓第二偵以後參考點
            self.distance = np.linalg.norm(series[0] - series[1])*0.5
            self.next_reference_point = series[0]
            self.previous_point = self.next_reference_point
            return self.next_reference_point#回傳下一個參考點
    
    def roll_image_points(self, image_points):
        self.image_points = np.array(image_points, dtype=np.float32)#轉NumPy格式

        if self.isFirstFrame:#第一偵抓第一個參考點
            self.__find_reference_point(self.image_points)#放入第一偵參考點
            self.isFirstFrame = False

        elif self.isFirstFrame == True:#第二偵開始  
            previous_point = self.next_reference_point#前一偵的點
            cv2.circle(self.img,previous_point.astype(int),self.distance.astype(int),(255,0,255),2)
            counter = 0
            #旋轉直到下個imgPoints的參考點有在前一個imgPoints的參考點範圍內，就break
            while True:
                counter+=1
                next_point = self.__find_reference_point(self.image_points)#下一偵的點
                length = np.linalg.norm(previous_point - next_point)#算前後點的距離
                if length <= self.distance or counter == 5:
                    break
                image_points = np.roll(image_points, -1, axis=0)

        image_points = [tuple(point) for point in image_points]#轉成pnp需要的格式，並回傳
        return image_points
        
if __name__ == "__main__":
    pass
