import cv2
file =  ''
pic = cv2.imread(file , 0)
height, width = pic.shape
height  , width = pic.shape

# 遍历像素点 (3*3观察)
for i in range(0+1,height-1):
    for j in range(0+1,width-1):
        # 该点的位置
        point_place__temp = (i,j)
        pic[i , j] =
