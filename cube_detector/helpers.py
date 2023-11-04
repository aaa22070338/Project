from scipy.spatial import cKDTree
import numpy as np

def correct_coordinates(approx_points_pack, epsilon):
    correct_approx_points_pack = []
    # 取出所有端點座標
    all_points = np.vstack(
        [point for points in approx_points_pack for point in points]
    )
    # 做兩次凝聚相鄰座標點
    all_points = merge_close_points(all_points, 3.5 * epsilon, iter=2)
    if all_points is None:
        return None,None
    update_points = set(tuple(point) for point in all_points)
    # 更新舊多邊形端點座標成距離最近的凝聚座標點，並且紀錄有被更新到的座標點
    for approx_points in approx_points_pack:
        for i in range(len(approx_points)):
            target_point = np.array(approx_points[i])
            distances = np.linalg.norm(all_points - target_point, axis=1)
            nearest_index = np.argmin(distances)
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