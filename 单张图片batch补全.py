# trans gray pic : black to white ,or white to black
import cv2

originpic = '01.png'

pic = cv2.imread(originpic, 0)

pic_copy  = pic.copy()

# 是彩色图转换为黑白
if len(pic_copy)==3 : src_RGB = cv2.cvtColor(pic_copy, cv2.COLOR_RGB2GRAY)
height , width = pic_copy.shape

def find_the_point_to_inpaint(origin,mask):
    import numpy as np
    flag = np.zeros(shape=origin.shape)
    height, width = mask.shape
    for i in range(height-1):
        for j in range(width-1):
            if mask[i,j] == 255

