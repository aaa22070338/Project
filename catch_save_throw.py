import json
import numpy as np
class CatchAndSave:
    def __init__(self):
        CatchAndSave.isclear = False
        CatchAndSave.completed_Save = False
        CatchAndSave.count = 0

    @classmethod
    def color_name(cls):
        cls.color_name = ['green', 'red', 'yellow', 'purple' ,'blue'] 

    @classmethod
    def reset(cls):
        cls.isclear = False
        cls.completed_Save = False
        cls.count = 0

    @classmethod
    def catch_save(cls ,color_name, x:list , y:list ,rz:list, input_count ):
        cls.input_count = input_count
        if cls.completed_Save == False:
            if cls.isclear == False:
                with open("input_file_a.json", 'w') as file:
                    file.seek(0)
                    file.truncate()
                    file.write('[') 
                cls.isclear = True
            color_coordinates_count = sum(1 for line in open("input_file_a.json"))-1
            #print(color_coordinates_count)
            if color_coordinates_count < input_count:
                data = {
                    "color_name": color_name,
                    "x": x,
                    "y": y,
                    "rz": rz
                }
                with open("input_file_a.json", 'a+') as file:  # 打开文件追加模式
                    if color_coordinates_count  ==0:
                        file.write('\n')
                    file.write('\t')    
                    if color_coordinates_count > 0:
                        file.write(',')
                    json.dump(data, file)  # 将坐标数据写入文件
                    file.write('\n')    
                cls.count += 1

            if color_coordinates_count >= cls.input_count:
                with open("input_file_a.json", 'a') as file:
                    file.write(']')
                cls.completed_Save = True

    
    @classmethod
    def coordinates(cls, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        list_x = [item["x"][0] for item in data]
        list_y = [item["y"][0] for item in data]
        list_rz = [item["rz"][0] for item in data]
        list_T =np.array( [list_x , list_y , list_rz ]).T
        median_value = np.median(list_T , axis=0)
        lower_limit = median_value - 3
        upper_limit = median_value + 3
        data_b = np.unique(np.clip(list_T, lower_limit, upper_limit), axis=0)
        average = np.mean(data_b ,axis = 0)
        return average

    # @classmethod
    # def coordinates_x(cls, file_path):
    #     with open(file_path, 'r') as file:
    #         data = json.load(file)
    #     list_x = [item["x"][0] for item in data]
    #     median_value = statistics.median(list_x)
    #     average_x = statistics.mean(value for value in list_x if median_value - 3 <= value <= median_value + 3)
    #     return average_x
    
    # @classmethod
    # def coordinates_y(cls, file_path):
    #     with open(file_path, 'r') as file:
    #         data = json.load(file)
    #     list_y = [item["y"][0] for item in data]
    #     median_value = statistics.median(list_y)
    #     average_y = statistics.mean(value for value in list_y if median_value - 3 <= value <= median_value + 3)
    #     return average_y
    
    # @classmethod
    # def coordinates_rz(cls, file_path):
    #     with open(file_path, 'r') as file:
    #         data = json.load(file)
    #     list_rz = [item["rz"][0] for item in data]
    #     median_value = statistics.median(list_rz)
    #     average_rz = statistics.mean(value for value in list_rz if median_value - 3 <= value <= median_value + 3)
    #     return average_rz




