'''
created by Y.Huang
2019.12.04
提取图片1和图片2的sift特征，并保存每个点的特征和坐标信息，留到main.py中处理
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 测试图片如下所示
for i in range(0,19+1,1):

    pic01 = 'try//0004_01.pngori_'+str(i)+'.png'
    pic02 = 'try//0004_01.pngori_'+str(i+1)+'.png'
    # 阈值越大，找到的特征点越少
    Threshold = 0.1


    # 读取图片1
    img = cv2.imread(pic01)
    # 缩放0.2倍，不然图像显示太大了
    # img = cv2.resize(img,None,fx=0.2,fy=0.2)
    # 读取图片2

    img1 = cv2.imread(pic02)

    # 把图片2缩放成图片1的大小，因为后面要把两张图片拼起来画对应特征点的连线
    # img1 = cv2.resize(img1,(img.shape[1],img.shape[0]))
    # print(img.shape,img1.shape)

    # 处理成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 下面是提取sift特征
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=Threshold)
    # 阈值越大，找到的特征点越少

    # sift1 = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.1)

    # kp是特征点（包括坐标信息），ds相当于特征点的特征（128维）
    kp,ds= sift.detectAndCompute(gray, None)
    kp1,ds1= sift.detectAndCompute(gray1, None)
    # kp1 = sift1.detect(gray1,None)

    cv2.drawKeypoints(image=img,outImage=img,
                      keypoints=kp,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                      color=(51, 163, 236))
    cv2.drawKeypoints(image=img1,outImage=img1,
                      keypoints=kp1,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                      color=(0, 0, 255))


    cv2.imshow('0', img)
    cv2.imshow('1', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 下面是保存信息的部分

    # 保存特征点的坐标
    coor = []
    for i in range(ds.shape[0]):
        coor.append(kp[i].pt)

    coor = np.array(coor)
    print(coor.shape)
    # 保存特征
    np.save('img'+str(i+1)+'_fea.npy',ds)

    # 保存坐标
    np.save('img+'+str(i+1)+'_coor.npy',coor)

    coor1 = []
    for i in range(ds1.shape[0]):
        coor1.append(kp1[i].pt)

    coor1 = np.array(coor1)
    print(coor1.shape)
    np.save('img'+str(i+1)+'_fea.npy',ds1)
    np.save('img'+str(i+1)+'_coor.npy',coor1)