import  numpy as np
import  cv2
# 0值为背景，1为前景
# Zhang-Suen细化算法的整个迭代过程分为两步：
#
# Step One：循环所有前景像素点，对符合如下条件的像素点标记为删除：
#
# 1.      2 <= N(p1) <=6，中心像素p1周围的目标像素（二值中的1）的个数在2~6之间；
#
# 2.      S(P1) = 1，8邻域像素中，按顺时针方向，相邻两个像素出现0→1的次数；
#
# 3.      P2 * P4 * P6 = 0
#
# 4.      P4 * P6 * P8 = 0

# 定义像素点周围的8邻域
#                P9 P2 P3
#                P8 P1 P4
#                P7 P6 P5
# 9个点的排布
# 定义像素点周围的8邻域
    #                P9 P2 P3
    #                P8 P1 P4
    #                P7 P6 P5

def neighbours(x, y, image):
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1,y], img[x_1,y1], img[x,y1], img[x1,y1],  # P2,P3,P4,P5
            img[x1,y], img[x1,y_1], img[x,y_1], img[x_1,y_1]]  # P6,P7,P8,P9


# 计算邻域像素从0变化到1的次数
def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # P2,P3,...,P8,P9,P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3),(P3,P4),...,(P8,P9),(P9,P2)


# Zhang-Suen 细化算法
def zhangSuen(image):
    Image_Thinned = image.copy()  # Making copy to protect original image
    changing1 = changing2 = 1
    while changing1 or changing2:  # Iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x,y] == 1 and  # Condition 0: Point P1 in the object regions
                        2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                        transitions(n) == 1 and  # Condition 2: S(P1)=1
                        P2 * P4 * P6 == 0 and  # Condition 3
                        P4 * P6 * P8 == 0):  # Condition 4
                    changing1.append((x, y))
        for x, y in changing1:
            Image_Thinned[x,y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x,y] == 1 and  # Condition 0
                        2 <= sum(n) <= 6 and  # Condition 1
                        transitions(n) == 1 and  # Condition 2
                        P2 * P4 * P8 == 0 and  # Condition 3
                        P2 * P6 * P8 == 0):  # Condition 4
                    changing2.append((x, y))
        for x, y in changing2:
            Image_Thinned[x,y] = 0
    return Image_Thinned
def convert_boolen_number(array):
    import numpy as np
    array = np.array(array)
    array_number = np.zeros(shape=array.shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j] == True:
                array_number[i][j] = 255
    return array_number
if __name__ == '__main__':
    # 导入库
    import matplotlib
    import matplotlib.pyplot as plt
    import skimage.io as io

    # 将图像转为灰度图像
    from PIL import Image
    pic_place = "01.png"
    img = Image.open(pic_place).convert('L')
    pic_place_gray = pic_place.replace('.png','gray'+'.png')
    img.save(pic_place_gray)

    # 读取灰度图像
    Img_Original = io.imread(pic_place_gray)

    # 对图像进行预处理，二值化
    from skimage import filters
    from skimage.morphology import disk

    # 中值滤波
    Img_Original = filters.median(Img_Original, disk(5))
    # 二值化
    BW_Original = Img_Original < 235
    ########################################################
    # 对染色体图像应用Zhang-Suen细化算法
    BW_Skeleton = zhangSuen(BW_Original)
    import numpy as np


    BW_Skeleton = convert_boolen_number(BW_Skeleton)
    # import cv2
    # cv2.imshow('thin is :',BW_Skeleton)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 生成骨架图像并保存
    print(BW_Skeleton)
    Skeleton = np.ones((BW_Skeleton.shape[0], BW_Skeleton.shape[1]), np.uint8) * 255  # 生成一个空灰度图像
    BW_Skeleton = BW_Skeleton + 0
    for i in range(BW_Skeleton.shape[0]):
        for j in range(BW_Skeleton.shape[1]):
            if BW_Skeleton[i][j] == 0:
                Skeleton[i][j] = 0

    plt.axis('off')
    plt.imshow(Skeleton, cmap=plt.cm.gray)

    import imageio

    imageio.imwrite('straight//3Skeleton.png', Skeleton)
