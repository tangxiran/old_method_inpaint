# https://blog.csdn.net/wsp_1138886114/article/details/82872838?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.channel_param
import  cv2
import numpy as np
def median_blur_demo(image):
    # 中值模糊  对椒盐噪声有很好的去燥效果
    dst = cv2.medianBlur(image, 5)
    cv2.imshow("median_blur_demo", dst)
    return dst


def blur_demo(image):
    """
	均值模糊 : 去随机噪声有很好的去噪效果
	（1, 15）是垂直方向模糊，（15， 1）是水平方向模糊
	"""
    dst = cv2.blur(image, (1, 15))
    cv2.imshow("avg_blur_demo", dst)
    return dst

def custom_blur_demo(image):
    """
	用户自定义模糊
	下面除以25是防止数值溢出
	"""
    kernel = np.ones([5, 5], np.float32)/25
    dst = cv2.filter2D(image, -1, kernel)
    cv2.imshow("custom_blur_demo", dst)
    return dst

# 边缘保留滤波EPF
'''
进行边缘保留滤波通常用到两个方法：高斯双边滤波和均值迁移滤波。

双边滤波（Bilateral filter）是一种非线性的滤波方法，是结合图像的空间邻近度和像素值相似度的一种折中处理，同时考虑空域信息和灰度相似性，达到保边去噪的目的。
双边滤波器顾名思义比高斯滤波多了一个高斯方差 σ － d \sigma－dσ－d，它是基于空间分布的高斯滤波函数，所以在边缘附近，离的较远的像素不会太多影响到边缘上的像素值，这样就保证了边缘附近像素值的保存。但是由于保存了过多的高频信息，对于彩色图像里的高频噪声，双边滤波器不能够干净的滤掉，只能够对于低频信息进行较好的滤波
双边滤波函数原型：
'''
def bi_demo(image):
    #双边滤波
    dst = cv2.bilateralFilter(image, 0, 100, 5)
    cv2.imshow("bi_demo", dst)
    return dst


def gaussian_blur(image):
    dst = cv2.GaussianBlur(image,(15,15),0)
    cv2.imshow('gaussian_blur' , dst)
    return dst

def shift_demo(image):
    #均值迁移，
    dst = cv2.pyrMeanShiftFiltering(image, 10, 50)
    cv2.imshow("shift_demo", dst)
    return dst

if __name__ == '__main__':
    file = 'data//0001_01_0'
    # for i in range(1,100+1,1):
    file = file+str(0)+'.png'
    file = 'data//22.png'
    pic = cv2.imread(file)
    cv2.imshow("origin", pic)
    dst  = median_blur_demo(pic)
    # cv2.imshow("median_blur_demo", dst)
    dst = blur_demo(pic)

    dst =custom_blur_demo(pic)

    dst = bi_demo(pic )

    dst = shift_demo(pic)
    dst =gaussian_blur(pic )
    cv2.waitKey(0)
    cv2.destroyAllWindows()