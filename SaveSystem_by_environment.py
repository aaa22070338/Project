import numpy as np

class save_system:
    def __init__(self, series) -> None:
        self.series = series
        for i in range(len(self.series)):
            setattr(self, f"color_{i}", False)

    isclear = False
    completed_Save = False
    times = 0

    # @classmethod
    def reset(self):
        self.isclear = False
        self.completed_Save = False
        self.times = 0

    # @classmethod
    def save_coordinate_by_color(self, color_name, x, y, input_count):
        if self.completed_Save == False:
            if self.isclear == False:
                #清除檔案
                with open("SaveCoor_environment.txt", 'r+') as Clear:
                    Clear.seek(0)
                    Clear.truncate() 
                self.isclear = True
            #檢查儲存次數有沒有達到上限(input_count)
            color_coordinates_count = sum(1 for line in open("SaveCoor_environment.txt") if line.startswith(color_name))
        
            if color_coordinates_count < input_count:
                data = np.array([x, y])
                data_with_newline = np.vstack([data, np.zeros_like(data[0])])
                data_with_newline = data_with_newline.T

                header_str = f"{color_name} {data_with_newline[0, 0]:.2f}\t{data_with_newline[0, 1]:.2f}\t{data_with_newline[0, 2]:.2f}"

                with open("SaveCoor_environment.txt", 'a') as file:
                    file.write(header_str)
                    file.write('\n')
                self.times += 1
            
            # if(len(self.color_set) == len(self.series)):
            #     self.completed_Save = True
            
            for i in range(len(self.series)):
                if (sum(1 for line in open("SaveCoor_environment.txt") if line.startswith(self.series[i])) >= input_count):
                    setattr(self, f"color_{i}", True)

            if all(getattr(self, f"color_{i}") for i in range(len(self.series))):
                self.completed_Save = True
            # if (sum(1 for line in open("SaveCoor_environment.txt") if line.startswith("green")) >= input_count):
            #     self.current_count += 1
            # # 遍歷每一行，計算行數
            # with open("SaveCoor_environment.txt", 'r') as file:
            #     line_count = sum(1 for line in file)
            # # 檢查是否有足夠的行數符合條件 放4個方塊
            # if line_count >= input_count * 4:
            #     self.completed_Save = True

        
    def get_coordinates_by_color(self, colors: list):
        with open("SaveCoor_environment.txt", 'r') as read:
            lines = read.readlines()
        all_coordinates = []
        for color in colors:
            coordinates = []
            current_color = color

            for line in lines:
                if line.startswith(current_color):
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


            # # 顏色列表，檢查每種顏色的行數是否符合條件##############################################重改
            # for color in color_name:
            #     line_count = sum(1 for line in open("SaveCoor_environment.txt") if line.startswith(color))
            #     # 如果該顏色的行數符合條件，則增加通過的顏色數量
            #     if line_count >= input_count * 0.7:
            #         valid_color_count += 1
            # # 檢查是否所有顏色都通過檢查
            # if valid_color_count == len(color_name):
            #     self.completed_Save = True