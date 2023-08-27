import cv2
import numpy as np

# 創建一個黑色背景
image = np.zeros((500, 500, 3), dtype=np.uint8)

# 定義兩個點的座標
point1 = (100, 100)
point2 = (300, 300)

# 繪製連接兩點的線段
cv2.line(image, point1, point2, (0, 255, 0), 2)

# 判斷兩點之間是否有線段連接
def are_points_connected(img, pt1, pt2):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.pointPolygonTest(contour, pt1, False) >= 0 and cv2.pointPolygonTest(contour, pt2, False) >= 0:
            return True, contour
    return False, None

connected, contour = are_points_connected(image, point1, point2)

if connected:
    print("Points are connected by a line.")
    cv2.drawContours(image, [contour], 0, (0, 0, 255), 1)  # 將輪廓繪製為紅色
else:
    print("Points are not connected by a line.")

# 顯示圖像
cv2.imshow("Image with Line and Contour", image)
cv2.waitKey(0)
cv2.destroyAllWindows()