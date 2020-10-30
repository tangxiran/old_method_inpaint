'''
created by Y.Huang
2019.12.04
'''
import numpy as np

import hnswlib

import cv2

# 这是一个颜色列表，为了画连线时每个线条颜色不同
color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
# 读取在sift.py保存的信息
fea1 = np.load('img1_fea.npy')
fea2 = np.load('img2_fea.npy')
pt1 = np.load('img1_coor.npy')
pt2 = np.load('img2_coor.npy')

# 下面是创新，利用hnsw搜索算法，在图片2里找到每一个特征点在图片1中距离最近的2个
p = hnswlib.Index(space='l2', dim=fea1.shape[1])
p.init_index(max_elements=2000, ef_construction=100, M=10)
p.set_ef(16)
p.set_num_threads(4)
hnsw_label = list(range(0, fea1.shape[0]))
# 将图片1的所有特征加入hnsw结构
p.add_items(fea1, hnsw_label)

# 图片2中的特征点当作probe，搜索出它的前2个最近邻
matched = []
for i in range(fea2.shape[0]):
    knn_label, dis = p.knn_query(fea2[i], k=2)
    # 如果第一个最近邻小于第二最近邻×0.3,就把这个特征点当作配准点，加入到matched中
    if dis[0][0]< 0.3*dis[0][1]:
    # if dis[0][0]/dis[0][1] < 0.3:
        matched.append([pt1[knn_label[0][0]],pt2[i]])


print(len(matched))
#######################
# 组合图像画线

# 注意这里读取图片的操作要和sift.py中一样，不然画出来会错位
img = cv2.imread('img3.jpg')
img = cv2.resize(img,None,fx=0.2,fy=0.2)
img1 = cv2.imread('img4.jpg')
img1 = cv2.resize(img1,(img.shape[1],img.shape[0]))
print(img.shape,img1.shape)

# 为了在后面拼接图像的时候看出区别，我给两张图片加了黄色的外边框
cv2.rectangle(img,(0,0),(img.shape[1],img.shape[0]),(51, 163, 236),2)
cv2.rectangle(img1,(0,0),(img1.shape[1],img1.shape[0]),(51, 163, 236),2)
# 两张图片拼成一张，左边和右边
new_img = np.hstack((img,img1))

'''        图片1的点    图片2的点
matched=[ [[x0,y0]  ,  [x1,y1]],  每一行代表匹配上的点
          [[x0',y0'],  [x1',y1']],...
          [[..]     ,  [..]]
        ]

'''
# 根据matched画连线
for i in range(len(matched)):
    index = i%6
    # print(index)
    cv2.line(new_img,(int(matched[i][0][0]), int(matched[i][0][1])),
             (img.shape[1]+int(matched[i][1][0]), int(matched[i][1][1])),
             color[index],
             2)
    cv2.circle(new_img,(int(matched[i][0][0]), int(matched[i][0][1])),3,(51, 163, 236),-1)
    cv2.circle(new_img,(img.shape[1]+int(matched[i][1][0]), int(matched[i][1][1])),3,(51, 163, 236),-1)

cv2.imshow('',new_img)
cv2.waitKey(0)

########################
##拼接图像
src_pts = []
dst_pts = []
for i in range(len(matched)):
    src_pts.append(matched[i][0])
    dst_pts.append(matched[i][1])

src_pts = np.array(src_pts)
dst_pts = np.array(dst_pts)
print(src_pts)
print(dst_pts)
H=cv2.findHomography(src_pts,dst_pts)

h,w=img.shape[:2]
h1,w1=img1.shape[:2]
shft=np.array([[1.0,0,w],[0,1.0,0],[0,0,1.0]])
M=np.dot(shft,H[0])            #获取左边图像到右边图像的投影映射关系
dst_corners=cv2.warpPerspective(img,M,(w*2,h))#透视变换，新图像可容纳完整的两幅图
cv2.imshow('tiledImg1',dst_corners)   #显示，第一幅图已在标准位置
dst_corners[0:h,w:w*2]=img1
#cv2.imwrite('tiled.jpg',dst_corners)
cv2.imshow('tiledImg',dst_corners)
cv2.waitKey(0)





