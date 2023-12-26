from typing import Literal, TypeAlias
import cv2
import numpy as np
from ultralytics import YOLO
import copy
from cube_detector.helpers import *

ColorType : TypeAlias = Literal['red','blue','green','yellow','white','orange','purple']
DetectMethod : TypeAlias = Literal['tracking']
class CubeDetector:
    def __init__(self, cube_model:YOLO,cube_surface_model:YOLO) -> None:
        self.model = cube_model
        self.suface_model = cube_surface_model
        self.is_first_frame = False
        self.tracking_props: dict[ColorType, tracker] = {}

    def detect(self, img:cv2.UMat, index:int|None=None, color:ColorType | None= None, show_process_img=False,show_text=True,method:DetectMethod|None=None):
        self.img = img
        self.output_img = img.copy()
        self.finding_color = color
        self.detect_method = method
        self.cube_image_points: dict[ColorType,np.ndarray] = {}
        self.cube_contour_outer: dict[ColorType,np.ndarray] = {}
        self.process_color : ColorType | None = None

        self.show_img_process = show_process_img
        self.show_text = show_text

        cube_results = self.model(self.img,verbose = False)[0]

        if cube_results.boxes is None or cube_results.masks is None:
            print("cube not found")
            return self.output_img
        
        if index is not None:
            box = cube_results.boxes[index]
            mask = cube_results.masks[index]
            self.__detect_process(mask,box)
            return self.output_img
            
        for box, mask in zip(cube_results.boxes, cube_results.masks):
            self.__detect_process(mask,box)
        return self.output_img
    
    def __detect_process(self,mask,box):
        try:
            # 圖片處理
            masked_image = self.__cube_object_detect(mask, box)
            color_rgb = self.__color_detect(masked_image)
            if self.finding_color != None and not (self.process_color == self.finding_color):
                raise DetectError("未偵測到所要搜尋顏色")
            corner_image,corner_points =  self.__conner_detect(masked_image, color_rgb)
            plane_corners = self.__largest_plane_detect(corner_image,corner_points)

            # 儲存處理後狀態
            if box.conf.cpu() > 0.75:
                self.cube_image_points[self.process_color]=plane_corners
                self.cube_contour_outer[self.process_color]=self.contour_outer
                # tracking_origin_point= plane_corners[0]
                # tracking_range = 0.5 * np.linalg.norm(plane_corners[0]-plane_corners[1])
                # self.tracking_props[self.process_color] = (tracking_origin_point,tracking_range) # 原點追跡，第一項紀錄原點位置，第二項紀錄掃秒範圍
        except DetectError as Error:
            print(Error)

    def get_cube_sequence_imagePoints(self,color:ColorType):
        return self.cube_image_points.get(color, None)
    
    def get_cube_contour_outer(self,color:ColorType):
        return self.cube_contour_outer.get(color, None)
    
    def __cube_object_detect(self, mask, box):
        height,width = self.img.shape[:2]
        
        contour_outer = np.array(mask.xyn).reshape(-1,2)
        contour_outer[:,0]=contour_outer[:,0]*width
        contour_outer[:,1]=contour_outer[:,1]*height
        segment = np.zeros((height,width), dtype=np.uint8)
        if np.any(np.isnan(contour_outer)) or len(contour_outer)==0:
            raise DetectError("[方塊]外輪廓偵測失敗")
        segment = cv2.drawContours(segment,[contour_outer.astype(int)],-1, 255, -1)
        masked_image = cv2.bitwise_and(self.img, self.img, mask=segment)
        self.contour_outer=contour_outer
        return masked_image

    def __color_detect(self, masked_image):
        hsv_ranges = {
            "red": np.array([[0, 222, 0], [20, 255, 255]], dtype=np.uint8),
            "yellow": np.array([[8, 101, 74], [105, 255, 255]], dtype=np.uint8),
            "green": np.array([[50, 101, 0], [112, 255, 108]], dtype=np.uint8),
            # "blue": np.array([[0, 0, 0], [179, 255, 18]], dtype=np.uint8),
            "purple": np.array([[83, 31, 0], [161, 160, 145]], dtype=np.uint8)
        }
        color_dict = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 128, 0),
            # "blue": (255, 0, 0),
            "purple": (160, 0, 160)
        }
        hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
        maximum_pixels_color = None
        pixel_counts = {}
        for color_name, hsv_range in hsv_ranges.items():
            lower_bound = hsv_range[0]
            upper_bound = hsv_range[1]

            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            pixel_counts[color_name] = cv2.countNonZero(mask)
        maximum_pixels_color = max(pixel_counts, key=pixel_counts.get)
        if maximum_pixels_color is None:
            raise DetectError("無法判斷顏色")
        self.process_color = maximum_pixels_color
        return color_dict[maximum_pixels_color]

    def __conner_detect(self, masked_image, color_rgb):
        result2 = self.suface_model(self.img,verbose = False)
        masks = result2[0].masks
        boxes = result2[0].boxes
        if masks is None :
            raise DetectError("[面]輪廓模型偵測失敗")
        isClosed = True
        self.epsilon_outer = 0.015 * cv2.arcLength(self.contour_outer, isClosed)
        contours=[self.contour_outer.copy().astype(int)]
        approx_points_pack = [cv2.approxPolyDP(self.contour_outer,self.epsilon_outer,isClosed)]
        for mask,box in zip(masks,boxes):
            if box.conf.cpu() <0.8:
                continue
            contour = np.array(mask.xyn).reshape(-1,2)
            contour[:,0]=contour[:,0]*masked_image.shape[1]
            contour[:,1]=contour[:,1]*masked_image.shape[0]
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
        
        if len(approx_points_pack)==1 or np.any(approx_points_pack is None):
            raise DetectError("[面]輪廓抓取近似多邊形失敗")
        # 將打包的多邊形端點整理校正位置，首項為校正後的多邊形，次項為校正後的各個座標點
        correct_approx_points_pack, updated_points = correct_coordinates(
            copy.deepcopy(approx_points_pack), epsilon=self.epsilon_outer
        )
        if correct_approx_points_pack is None:
            raise DetectError("[面]輪廓聚類失敗")

        updated_points = np.float32(list(updated_points))
        correct_approx_image = np.zeros(
            (masked_image.shape[0], masked_image.shape[1]), dtype=np.uint8
        )
        for correct_approx_points in correct_approx_points_pack:
            correct_approx_image = cv2.polylines(
                correct_approx_image, [correct_approx_points.astype(int)], isClosed, (255), 2
            )

        for point in updated_points:
            x,y=np.intp(point)
            cv2.circle(self.output_img, (x, y), 5, color_rgb, -1)


        if self.show_img_process:
            contour_image = np.zeros(
                (masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8
            )
            contour_image = cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 1)

            approx_image = np.zeros(
                (masked_image.shape[0], masked_image.shape[1]), dtype=np.uint8
            )
            for approx_points in approx_points_pack:
                approx_image = cv2.polylines(approx_image, [approx_points.astype(int)], isClosed, (255), 2)
            for point in updated_points:
                x, y = np.intp(point)
                cv2.circle(masked_image, (x, y), 5, color_rgb, -1)
            
            cv2.imshow("origin", self.output_img)
            cv2.imshow("masked", masked_image)
            cv2.imshow("contour", contour_image)
            cv2.imshow("approx", approx_image)
            cv2.imshow("correct approx", correct_approx_image)
        return correct_approx_image,updated_points
    
    def __largest_plane_detect(self,corner_image,corner_points):
        if corner_image is None or corner_points is None:
            # return None
            raise DetectError("[方塊角點]抓取失敗")
        # 抓修正後的方塊圖，掃描內外輪廓，去除外輪廓得到各個面的面輪廓，並依據面積大小進行排序
        contours,_ = cv2.findContours(corner_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1:]
        contours = sorted(contours,key=cv2.contourArea,reverse=True)
        # 抓出近似四邊形後的輪廓
        contours_approx = [contour_approx for contour_approx in [cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True) for contour in contours ] if len(contour_approx)==4]
        # 確認面輪廓是否存在
        if len(contours_approx) == 0:
            raise DetectError("無近似四邊形的輪廓")

        # 若只有一個四角面輪廓則使用四點座標標定法，否則六點座標標定
        draw_squence_points = None
        # tracking_props = self.tracking_props.get(self.process_color,None)
            

        _ , contour_touch_points = correct_coordinates(copy.deepcopy(contours_approx),self.epsilon_outer)
        if contour_touch_points is not None and len(contour_touch_points)>=1:
            contour_touch_points = np.array([point for point in contour_touch_points])
            # if (tracking_props is not None and
            #     (origin_point:=move_to_closest_point(contour_touch_points,tracking_props[0],tracking_props[1])) is not None):
            #     origin_point=origin_point
            # else:
            #     origin_point=contour_touch_points[0]
            origin_point=contour_touch_points[0]

            # origin_point = tracking_point if tracking_point is not None else contour_touch_points[0]
            draw_squence_points= self.__grab_six_points(contours_approx,origin_point)
        if draw_squence_points is not None:
            if self.show_text:
                for i , point in enumerate(draw_squence_points):
                    cv2.putText(self.output_img, f"{i}", list(np.intp(point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            # 真正的角點為前面抓角點處理所得到的角點，所以將紀錄座標更新成最接近的角點
            closest_indices = np.argmin(np.linalg.norm(draw_squence_points[:, np.newaxis, :] - corner_points, axis=2), axis=1)
            draw_squence_points = corner_points[closest_indices]
            return np.array(draw_squence_points)
        # 四點，面座標標定，當無法檢測到六點位置時使用

        draw_squence_points=[]
        # if (tracking_props is not None and
        #     (origin_point:=move_to_closest_point(contours_approx[0],tracking_props[0],tracking_props[1])) is not None):
        #     origin_point=origin_point
        # else:
            # origin_point=contours_approx[0][0]
        
        origin_point=contours_approx[0][0]

        coordinates=[(0,0,0),(0,1,0),(1,1,0),(1,0,0)]
        for point, coordinate in zip(contours_approx[0],coordinates):
            # 真正的角點為前面抓角點處理所得到的角點，所以將紀錄座標更新成最接近的角點
            point=min(corner_points, key= lambda corner_point: np.linalg.norm(corner_point-np.array(point)))
            draw_squence_points.append(point)
            # 將標點情況繪製於圖像，方便檢查
            if self.show_text:
                cv2.putText(self.output_img, f"{coordinate}", list(np.intp(point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        if self.show_img_process:
            contour_image = np.zeros(
                (corner_image.shape[0], corner_image.shape[1], 3), dtype=np.uint8
            )
            contour_image = cv2.drawContours(
                contour_image, contours, -1, (0, 0, 255), 1
            )
            if contours!=[]:
            # 顯示最大方塊塊輪廓面
                cv2.fillPoly(contour_image, [contours[0]], (255, 0, 0)) 
                contour_image = cv2.drawContours(
                    contour_image, contours, -1, (0, 0, 255), 1
                )
            cv2.imshow("contour", contour_image)
            # cv2.waitKey(0)
        draw_squence_points = np.array(draw_squence_points)
        return draw_squence_points
    

    def __grab_six_points(self,contours_approx, origin_point):
        # 創建空陣列準備紀錄座標順序
        draw_squence_points=np.empty((0,2))
        for contour in contours_approx:
            contour = contour.reshape(4,2)
            distances = np.linalg.norm(contour - origin_point, axis=1)
            # 透過距離找尋與輪廓接觸點接觸的輪廓
            if np.all(distances>5):
                continue
            # 找哪個點與輪廓接觸，找出他的索引，平移陣列順序使該接觸點作為原點
            target_index = np.where(distances<5)[0]
            contour = np.roll(contour, - target_index,axis=0)
            # 判斷座標順序紀錄狀況，若無則四點個作為座標，前四個點，若已經繪製四個點則紀錄倒數兩個點
            if draw_squence_points.shape[0]==0:
                draw_squence_points = np.vstack((draw_squence_points,contour))
            elif draw_squence_points.shape[0]==4:
                draw_squence_points = np.vstack((draw_squence_points,contour[2:,:]))
                break
        # 量測第二個點與第六個點是否座標一樣
        if len(draw_squence_points)<5:
            return None
        distance = np.linalg.norm(draw_squence_points[1]-draw_squence_points[5])
        if distance <10:
            # 一樣的話前四個左標點就順時針調換一次順序，後兩個改成抓該輪廓第二第三個座標點
            draw_squence_points[:4,:] = np.roll(draw_squence_points[:4,:],-1,axis=0)
            draw_squence_points[4:,:] = contour[1:3,:]                    
        return np.array(draw_squence_points)
    
class DetectError(Exception):
    def __init__(self, message):
        super().__init__(message)


if __name__ == "__main__":

    pass
