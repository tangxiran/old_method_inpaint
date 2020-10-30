# 特定的文件后缀保存
# def data_select(data_dir):  #
#     import  glob
#     file_list = list(glob.glob(data_dir + '/*.png')) + list(glob.glob(data_dir + '/*.jpg')) +list(glob.glob(data_dir + '/*.JPG'))   # get name list of all .png files
#     data = []
#     # print(file_list) # 得到文件的路径列表
#     return file_list
# if __name__ == '__main__':
#     dir = 'F:\数据集合\gray_origin'
#     data = data_select(dir)
#     print(data_select(dir))
#     for pic in data:

import  cv2
import numpy as np
pic = '01.png'
ss = cv2.imread(pic).astype(np.float32)
print(ss)