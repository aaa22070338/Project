import cv2
import numpy as np
img = cv2.imread("1012.jpg")

scaled = 600 / img.shape[0]
img = cv2.resize(img, None, fx=scaled, fy=scaled)

camera_matrix = np.array([[605.65, 0, 396.27],
                          [0, 605.15, 298.49],
                          [0, 0, 1]], dtype=np.float32)
dist_coeff = np.array([0.21287, -1.3059, 0.00038015, 0.00074224, 2.202], dtype=np.float32)

selected_points = np.array([(146, 578), (271, 580), (295, 519), (183, 424), (141, 463), (269, 461), (289, 423)], dtype=np.int32)
selected_points_3D = np.array([[[146, 578, 0], [271, 580, 0], [295, 519, 0], [183, 424, 0], [141, 463, 0], [269, 461, 0], [289, 423, 0]]], dtype=np.int32)

object_points = np.array([
    # [0, 0,   0],#1
    [0, 25,  0],#2
    [25, 25,  0],#3
    [25, 0,   0],#4
    [0, 0,  25],#5
    [0, 25, 25],#6
    [25, 25, 25],#7
    [25, 0,  25],#8
], dtype=np.float32)

image_points = np.array(selected_points, dtype=np.float32)

retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeff)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeff)
imgpts = imgpts.astype(np.int32)

img = cv2.line(img, (146, 578), tuple(imgpts[0].ravel()), (255,0,0), 5)
img = cv2.line(img, (146, 578), tuple(imgpts[1].ravel()), (0,255,0), 5)
img = cv2.line(img, (146, 578), tuple(imgpts[2].ravel()), (0,0,255), 5)

cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("imgpts:")
print(imgpts)