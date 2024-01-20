import numpy as np

class save_system:
    def __init__(self) -> None:
        pass
    isclear = False
    completed_Save = False
    count = 0

    @classmethod
    def reset(self):
        self.isclear = False
        self.completed_Save = False
        self.count = 0

    @classmethod
    def save_coordinate_by_color(self, color_name, x, y, Rz, input_count):
        if self.completed_Save == False:
            if self.isclear == False:
                #清除檔案
                with open("SaveCoor_grip.txt", 'r+') as Clear:
                    Clear.seek(0)
                    Clear.truncate() 
                self.isclear = True

            color_coordinates_count = sum(1 for line in open("SaveCoor_grip.txt") if line.startswith(color_name))
            
            if color_coordinates_count < input_count:
                data = np.array([x, y, Rz])
                data_with_newline = np.vstack([data, np.zeros_like(data[0])])
                data_with_newline = data_with_newline.T
                header_str = f"{color_name} {data_with_newline[0, 0]:.2f}\t{data_with_newline[0, 1]:.2f}\t{data_with_newline[0, 2]:.2f}"

                with open("SaveCoor_grip.txt", 'a') as file:
                    file.write(header_str)
                    file.write('\n')
                self.count += 1
            # if color_coordinates_count + 1 == 4:
            #     print(f"{color_name}，已達上限")

            if (sum(1 for line in open("SaveCoor_grip.txt") if line.startswith("green"))  == input_count):
                self.completed_Save = True
            if (sum(1 for line in open("SaveCoor_grip.txt") if line.startswith("yellow"))  == input_count):
                self.completed_Save = True
            if (sum(1 for line in open("SaveCoor_grip.txt") if line.startswith("red"))  == input_count):
                self.completed_Save = True
            if (sum(1 for line in open("SaveCoor_grip.txt") if line.startswith("purple"))  == input_count):
                self.completed_Save = True
        
    # def get_coordinates_by_color(self, colors: str):
    #     with open("SaveCoor_grip.txt", 'r') as read:
    #         lines = read.readlines()
    #     all_coordinates = []
    #     for color in colors:
    #         coordinates = []
    #         current_color = color

    #         for line in lines:
    #             if line.startswith(current_color):
    #                 x, y, z = map(float, line.strip(f"{color}").split('\t'))
    #                 coordinates.append((x, y, z))

    #         if coordinates:
    #             coordinates = np.vstack(coordinates)

    #         coordinates = self.remove_outlier(coordinates)
    #         all_coordinates.append(coordinates)
    #     return all_coordinates

    def get_coordinates_by_color(self, color: str):
        coordinates = []
        all_coordinates = []
        with open("SaveCoor_grip.txt", 'r') as read:
            for line in read:
                if not line.startswith(color):
                    continue

                x, y, z = map(float, line.strip(f"{color}").split('\t'))
                coordinates.append((x, y, z))
        if coordinates:
            coordinates = np.vstack(coordinates)

        coordinates = self.remove_outlier(coordinates)
        all_coordinates.append(coordinates)
        return all_coordinates    
    
    def remove_outlier(self, coordinates):
        # 計算每個維度的上下四分位距
        Q1 = np.percentile(coordinates, 25, axis=0)
        Q3 = np.percentile(coordinates, 75, axis=0)
        # 計算 IQR（上四分位距 - 下四分位距）
        IQR = Q3 - Q1
        # 定義極端值範圍
        outlier_range = 1 * IQR
        # 過濾掉極端值
        filtered_coordinates = [coord for coord in coordinates if np.all((Q1 - outlier_range) <= coord) and np.all(coord <= (Q3 + outlier_range))]
        for coord in filtered_coordinates:
            return coord


