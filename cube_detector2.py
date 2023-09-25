import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import cKDTree
import copy

def correct_coordinates(approx_points_pack, epsilon):
    correct_approx_points_pack = []
    # update_points = set()
    # 取出所有端點座標
    all_points = np.vstack(
        [point for points in approx_points_pack for point in points]
    )
    # 做兩次凝聚相鄰座標點
    all_points = merge_close_points(all_points, 3.5 * epsilon, iter=2)
    # print(all_points)
    if all_points is None:
        return None,None
    update_points = set(tuple(point) for point in all_points)
    # 更新舊多邊形端點座標成距離最近的凝聚座標點，並且紀錄有被更新到的座標點
    for approx_points in approx_points_pack:
        for i in range(len(approx_points)):
            target_point = np.array(approx_points[i])
            distances = np.linalg.norm(all_points - target_point, axis=1)
            nearest_index = np.argmin(distances)
            # if np.any(approx_points[i] != all_points[nearest_index]):
            #     update_points.add(tuple(all_points[nearest_index]))
            approx_points[i] = all_points[nearest_index]
        correct_approx_points_pack.append(approx_points)
    # 將校正過的多邊形同梱包，與有被更新到的座標返回
    return correct_approx_points_pack, update_points

def merge_close_points(coordinates, threshold_distance, iter):
    merged_points = []
    # 進行最近鄰查詢，找到接近的點
    for i in range(iter):
        if len(coordinates)==0:
            return None
        kdtree = cKDTree(coordinates)
        merged_points = []
        visited_points = set()
        for j, point in enumerate(coordinates):
            if j not in visited_points:
                # 找到與當前點接近的所有點
                close_indices = kdtree.query_ball_point(point, threshold_distance)
                visited_points.update(close_indices)

                # 計算接近的點的平均值，作為合併後的新點
                avg_x = sum(coordinates[idx][0] for idx in close_indices) / len(
                    close_indices
                )
                avg_y = sum(coordinates[idx][1] for idx in close_indices) / len(
                    close_indices
                )
                if i != 0 or len(close_indices) > 1:
                    merged_points.append((avg_x, avg_y))
        coordinates = merged_points
    return np.int0(merged_points)

class cubeDetector:
    def __init__(self, cube_model:YOLO,cube_surface_model:YOLO, isCudaSupport=False, processing_hegiht=300) -> None:
        self.model = cube_model
        self.suface_model = cube_surface_model
        self.isCudaSupport = isCudaSupport
        self.processing_height = processing_hegiht
        if isCudaSupport:
            self.sobelFilterX = cv2.cuda.createSobelFilter(0, cv2.CV_32F, 1, 0, 3)
            self.sobelFilterY = cv2.cuda.createSobelFilter(0, cv2.CV_32F, 0, 1, 3)
            self.cannyDetector = cv2.cuda.createCannyEdgeDetector(15, 100)
    
    def detect(self, img:cv2.UMat, index:int|None=None, color: str | None= None, show_process_img=False,show_text=True):
        self.img = img
        self.output_img = img.copy()
        self.cube_image_points={}
        self.cube_contour_outer={}
        cube_results = self.model(self.img)[0]
        # surface_results = self.suface_model(self.img)[0]
        if cube_results.boxes is None or cube_results.masks is None:
            print("cube not found")
            return self.output_img
        if index != None:
            box = cube_results.boxes[index]
            mask = cube_results.masks[index]
            masked_image = self.__cube_object_detect(mask, box)
            if masked_image is None:
                return self.output_img
            color_detected, color_rgb = self.__color_detect(masked_image)
            if color != None and not color_detected == color:
                print(f"index {index} is not {color}!")
                return self.output_img
            corner_image,corner_points = self.__conner_detect_process(masked_image, color_rgb, show_process_img)
            if corner_image is None:
                return self.output_img
            plane_corners = self.__largest_plane_detect(corner_image,corner_points,show_process_img,show_text)
            if not plane_corners is None:
                self.cube_image_points[color_detected]=plane_corners
        else:
            for box, mask in zip(cube_results.boxes, cube_results.masks):
                masked_image = self.__cube_object_detect(mask, box)
                if masked_image is None:
                    continue
                color_detected, color_rgb = self.__color_detect(masked_image)
                if color != None and not color_detected == color:
                    continue
                corner_image,corner_points =  self.__conner_detect_process(masked_image, color_rgb, show_process_img)
                if corner_image is None:
                    continue
                plane_corners = self.__largest_plane_detect(corner_image,corner_points,show_process_img,show_text)
                if not plane_corners is None:
                    self.cube_image_points[color_detected]=plane_corners
        return self.output_img
    
    def get_cube_largest_surface_imagePoints(self,color):
        return self.cube_image_points.get(color, None)
    def get_cube_contour_outer(self,color):
        return self.cube_contour_outer.get(color, None)
    
    def __cube_object_detect(self, mask, box):
        height,width = self.img.shape[:2]
        
        contour_outer = np.array(mask.xyn)
        contour_outer[:,:,0]=contour_outer[:,:,0]*width
        contour_outer[:,:,1]=contour_outer[:,:,1]*height
        segment = np.zeros((height,width), dtype=np.uint8)
        if np.any(contour_outer is None):
            return None
        segment = cv2.drawContours(segment,contour_outer.astype(int),-1, 255, -1)
        masked_image = cv2.bitwise_and(self.img, self.img, mask=segment)
        self.contour_outer=contour_outer
        # self.box_scale = self.processing_height / masked_image.shape[0]
        return masked_image

    def __color_detect(self, masked_image):
        hsv_ranges = {
            "white": np.array([[0, 15, 200], [75, 70, 255]], dtype=np.uint8),
            "red": np.array([[122, 63, 56], [132, 255, 255]], dtype=np.uint8),
            "orange": np.array([[100, 40, 170], [120, 255, 255]], dtype=np.uint8),
            "yellow": np.array([[80, 90, 100], [100, 255, 255]], dtype=np.uint8),
            "green": np.array([[50, 150, 60], [80, 210, 200]], dtype=np.uint8),
            "blue": np.array([[0, 70, 25], [40, 255, 210]], dtype=np.uint8),
        }

        color_dict = {
            "red": (0, 0, 255),  # 红色
            "orange": (0, 165, 255),  # 橘色
            "yellow": (0, 255, 255),  # 黄色
            "green": (0, 128, 0),  # 綠色
            "blue": (255, 0, 0),  # 藍色
            "white": (255, 255, 255),  # 白色
        }
        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
        max_color = None
        pixel_counts = {}
        for color_name, hsv_range in hsv_ranges.items():
            lower_bound = hsv_range[0]
            upper_bound = hsv_range[1]

            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            pixel_counts[color_name] = cv2.countNonZero(mask)

        max_color = max(pixel_counts, key=pixel_counts.get)  # type: ignore
        self.cube_contour_outer[max_color]=self.contour_outer
        return max_color, color_dict[max_color]



    def __conner_detect_process(self, masked_image, color_rgb, show_img_process):
        result2 = self.suface_model(self.img)
        masks = result2[0].masks
        boxes = result2[0].boxes
        if masks is None :
            return None,None
        isClosed = True
        self.epsilon_outer = 0.015 * cv2.arcLength(self.contour_outer, isClosed)
        contours=[self.contour_outer.copy().astype(int)]
        approx_points_pack = [cv2.approxPolyDP(self.contour_outer,self.epsilon_outer,isClosed)]
        for mask,box in zip(masks,boxes):
            if box.conf.cpu() <0.8:
                continue
            contour = np.array(mask.xyn)
            contour[:,:,0]=contour[:,:,0]*masked_image.shape[1]
            contour[:,:,1]=contour[:,:,1]*masked_image.shape[0]
            M = cv2.moments(contour)
            if M["m00"] != 0:
                centroid_X = int(M["m10"] / M["m00"])# 算形心x
                centroid_Y = int(M["m01"] / M["m00"])# 算形心y
            else:
                continue
            test_result = cv2.pointPolygonTest(self.contour_outer, (centroid_X, centroid_Y), False) # 計算重心是否位於方塊輪廓內部
            if test_result<=0:
                continue
            contours.append(contour.astype(int))
            epsilon = 0.025 * cv2.arcLength(contour,isClosed)
            approx_points = cv2.approxPolyDP(contour,epsilon,isClosed)
            approx_points_pack.append(approx_points)

        if show_img_process:
            contour_image = np.zeros(
                (masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8
            )
            # if np.any(contours is None):
            contour_image = cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 1)

        if len(approx_points_pack)==1 or np.any(approx_points_pack is None):
            return None,None
            
        if show_img_process:
            approx_image = np.zeros(
                (masked_image.shape[0], masked_image.shape[1]), dtype=np.uint8
            )
            for approx_points in approx_points_pack:
                approx_image = cv2.polylines(approx_image, [approx_points.astype(int)], isClosed, (255), 2)

        # 將打包的多邊形端點整理校正位置，首項為校正後的多邊形，次項為校正後的各個座標點
        correct_approx_points_pack, updated_points = correct_coordinates(
            approx_points_pack, epsilon=self.epsilon_outer
        )
        if correct_approx_points_pack is None:
            return None,None
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        # updated_points=cv2.cornerSubPix(gray,np.float32(list(updated_points)),(3,3),(-1,-1),criteria)
        updated_points = np.float32(list(updated_points))
        correct_approx_image = np.zeros(
            (masked_image.shape[0], masked_image.shape[1]), dtype=np.uint8
        )
        for correct_approx_points in correct_approx_points_pack:
            correct_approx_image = cv2.polylines(
                correct_approx_image, [correct_approx_points.astype(int)], isClosed, (255), 2
            )

        if show_img_process:
            for point in updated_points:
                x, y = np.intp(point)
                cv2.circle(masked_image, (x, y), 5, color_rgb, -1)

        for point in updated_points:
            x,y=np.intp(point)
            # x=int(point[0]/self.box_scale)+self.x1
            # y=int(point[1]/self.box_scale)+self.y1
            cv2.circle(self.output_img, (x, y), 5, color_rgb, -1)

        if show_img_process:
            cv2.imshow("origin", self.output_img)
            cv2.imshow("masked", masked_image)
            cv2.imshow("contour", contour_image)
            cv2.imshow("approx", approx_image)
            cv2.imshow("correct approx", correct_approx_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return correct_approx_image,updated_points
    
    def __largest_plane_detect(self,corner_image,corner_points,show_img_process=False,show_text=True):
        if corner_image is None or corner_points is None:
            return None

        contours,_ = cv2.findContours(corner_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1:]
        contours = sorted(contours,key=cv2.contourArea,reverse=True)

        contour_image = np.zeros(
                (corner_image.shape[0], corner_image.shape[1], 3), dtype=np.uint8
            )
        contour_image = cv2.drawContours(
            contour_image, contours, -1, (0, 0, 255), 1
        )

        contours = [contour for contour in contours if len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)) == 4]        
        contours_approx = [contour_approx for contour_approx in [cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True) for contour in contours ] if len(contour_approx)==4]
        # 抓相交角點
        if len(contours_approx)>1:
            _ , intersect_corners = correct_coordinates(copy.deepcopy(contours_approx),self.epsilon_outer)
            intersect_corners = np.array([[point] for point in intersect_corners])
            print(intersect_corners)
        # 判斷相交角點與近似輪廓的交點
            if len(intersect_corners)>1:
                for contour in contours_approx:
                    print(f"{len(contour)=}")
                    for point in intersect_corners:
                        distances = np.linalg.norm(contour.reshape(4,2) - point, axis=1)
                        nearest_index = np.argmin(distances)
                        print(f"{nearest_index=}")



        cube_coordinates=[]
        if len(contours_approx) == 0 :
            return None
        coordinates=[(0,0,0),(0,1,0),(1,1,0),(1,0,0)] if cv2.contourArea(contours_approx[0],True)>0 else [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]

        for point, coordinate in zip(contours_approx[0],coordinates):

            target_point = np.array(point)
            distances = np.linalg.norm(corner_points - target_point, axis=1)
            nearest_index = np.argmin(distances)
            point = corner_points[nearest_index]
            cube_coordinates.append(point)
            if show_text:
                cv2.putText(self.output_img, f"{coordinate}", list(np.intp(point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if contours!=[]:
            cv2.fillPoly(contour_image, [contours[0]], (255, 0, 0))  # 使用蓝色 (BGR格式)
            contour_image = cv2.drawContours(
                    contour_image, contours, -1, (0, 0, 255), 1
                ) 
        if show_img_process:
            cv2.imshow("contour", contour_image)
            # cv2.waitKey(0)
        cube_coordinates = np.array(cube_coordinates)
        return cube_coordinates


if __name__ == "__main__":
    pass
