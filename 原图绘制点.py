import os
import sys
import cv2

imagePath = "./img/"

image = cv2.imread(imagePath + "1.jpg")

point_size = 1
point_color = (0, 0, 255)  # BGR
thickness = 4  # 0 、4、8



height, width = pic.shape

# 遍历像素点 (3*3观察)
for i in range(0 + 1, height - 1):
    for j in range(0 + 1, width - 1):
        # 该点的位置
        point_place__temp = (i, j)
        # 判断是否为特征点 端点？交叉点？(交叉点还没考虑)
        value = pic[i, j]  # 该点的数值
        labels[i, j] = judgePoint(point_place__temp, pic=pic)

# 此处省略得到坐标的过程，coordinates存放坐标
# 格式为：coordinates=[[x1,y1],[x2,y2],[x3,y3],...,[xn,yn]]

for coor in coordinates:
    print(coor)
    cv2.circle(image, (int(coor[0]), int(coor[1])), point_size, point_color, thickness)

