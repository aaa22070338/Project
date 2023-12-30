import numpy as np

class SaveSystem:
    isclear = False
    completed_Save = False
    count = 0

    @classmethod
    def reset(self):
        self.isclear = False
        self.completed_Save = False
        self.count = 0

    @classmethod
    def save_coordinate(self, color_name, x, y, Rz):
        if self.completed_Save == False:
            if self.isclear == False:
                #清除檔案
                with open("SaveCoor.txt", 'r+') as Clear:
                    Clear.seek(0)
                    Clear.truncate() 
                self.isclear = True

            color_coordinates_count = sum(1 for line in open("SaveCoor.txt") if line.startswith(color_name))
            
            if color_coordinates_count < 4:
                data = np.array([x, y, Rz])
                data_with_newline = np.vstack([data, np.zeros_like(data[0])])
                data_with_newline = data_with_newline.T
                header_str = f"{color_name} {data_with_newline[0, 0]:.2f}\t{data_with_newline[0, 1]:.2f}\t{data_with_newline[0, 2]:.2f}"

                with open("SaveCoor.txt", 'a') as file:
                    file.write(header_str)
                    file.write('\n')
                self.count += 1
            if color_coordinates_count + 1 == 4:
                print(f"{color_name}，已達上限")

            if (sum(1 for line in open("SaveCoor.txt") if line.startswith("White")) and sum(1 for line in open("SaveCoor.txt") if line.startswith("Orange")) == 4):
                self.completed_Save = True

    def get_coordinate(line_number):
        with open("SaveCoor.txt", 'r') as read:
            lines = read.readlines()
            lines = lines[line_number - 1]
            values = lines.strip().split('\t')
            x = float(values[0])
            y = float(values[1])
            Rz = float(values[2])

            return x, y, Rz
        
    def get_coordinates_by_color(color: str):
        with open("SaveCoor.txt", 'r') as read:
            lines = read.readlines()

        coordinates = []
        current_color = color

        for line in lines:
            if line.startswith(current_color):
            #     current_color = color

            # if current_color == color:

                x, y, Rz = map(float, line.strip(f"{color}").split('\t'))
                coordinates.append((x, y, Rz))

        if coordinates:
            coordinates = np.vstack(coordinates)

        return coordinates