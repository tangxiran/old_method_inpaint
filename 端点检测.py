def getValue(point_temp,pic):
    import numpy as np
    import cv2
    i, j = point_temp
    return pic[i,j]

def drawFeaturePoint(pic_origin , coordinates, output_savePlace):
    import cv2
    import numpy as np
    # 点的尺寸
    point_size = 1
    point_color = (0, 0, 255)  # BGR,现在的是红色
    thickness = 0  # 0 、4、8

    for coor in coordinates:
        print(coor)
        # 横纵坐标的位置?
        cv2.circle(pic_color, (int(coor[1]), int(coor[0])), point_size, point_color, thickness)
    pic_todraw = output_savePlace
    # 在原图上标注出特征点,绘制在pic_todraw上
    cv2.imwrite(pic_todraw, pic_origin, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    # 胡绘制原图对比
    cv2.imshow('origin ', pic_origin)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return  0
def judgePoint(point_temp, pic ):
    # 按照插值寻找
    # 阈值1:与其共边的点的差距
    yuzhi01 = 60

    # 阈值2:与其临近的点的差距
    yuzhi02 = 100
    
    import numpy as np
    import cv2
    # 判断是否为特征点 ,
    i,j = point_temp
    # 判断要求： 周围8个点 ，共边的4个点有一个点与其差距小于yuzhi01，且有5个点的差距大于yuzhi02

    # 或者 比例 作为阈值
    if pic[i,j] < 100:return  0  # 特征点的值要大于100
    value =  pic[i,j]
    # 得到8个点的坐标信息
    for pppp in range(1):
        point1 = (i - 1, j - 1);
        point2 = (i - 1, j);
        point3 = (i - 1, j + 1);
        point4 = (i, j + 1);
        point5 = (i + 1, j + 1);
        point6 = (i + 1, j);
        point7 = (i + 1, j - 1);
        point8 = (i, j - 1);
    count_big = 0;    count_small = 0
    # 共边的四个点计算
    if abs(getValue(point2, pic) - value) < yuzhi01 or getValue(point2, pic) >= 100: count_big = count_big + 1
    if abs(getValue(point8, pic) - value) < yuzhi01 or getValue(point8, pic) >= 100: count_big = count_big + 1
    if abs(getValue(point6, pic) - value) < yuzhi01 or getValue(point6, pic) >= 100: count_big = count_big + 1
    if abs(getValue(point4, pic) - value) < yuzhi01 or getValue(point4, pic) >= 100: count_big = count_big + 1

    # if abs(getValue(point2, pic) - value) <yuzhi01  or getValue(point2, pic)>=100 :count_big = count_big + 1
    # if abs(getValue(point8, pic) - value) < yuzhi01 or getValue(point8, pic): count_big = count_big + 1
    # if abs(getValue(point6, pic) - value) < yuzhi01 or getValue(point6, pic): count_big = count_big + 1
    # if abs(getValue(point4, pic) - value )< yuzhi01 or getValue(point4, pic): count_big = count_big + 1
    # # 共边4点有两个及以上的点,满足差距很小,小于五十,说明  原始的该点处于线上
    if count_big>=2 :return  0

    if (value-getValue(point1,pic)) >=yuzhi02 : count_small = count_small + 1
    if (value-getValue(point2,pic)) >=yuzhi02 : count_small = count_small + 1
    if (value - getValue(point3, pic)) >= yuzhi02: count_small = count_small + 1
    if (value - getValue(point4, pic)) >= yuzhi02: count_small = count_small + 1
    if (value - getValue(point5, pic)) >= yuzhi02: count_small = count_small + 1
    if (value - getValue(point6, pic)) >= yuzhi02: count_small = count_small + 1
    if (value - getValue(point7, pic)) >= yuzhi02: count_small = count_small + 1
    if (value - getValue(point8, pic)) >= yuzhi02: count_small = count_small + 1
    #周围的8个点有5个及以上的数量与该中心点的差距大于yuzhi02 ,说明该点是断开的点

    if count_small>=5:return 1
    return 0

def judgePoint_bili(point_temp, pic ):
    #
    import numpy as np
    import cv2
    # 判断是否为特征点 ,
    i,j = point_temp
    # 判断要求： 周围8个点 ，共边的4个点有一个点与其差距小于50，且有5个点的差距大于80

    # 或者 比例 作为阈值
    if pic[i,j] < 100:return  0  # 特征点的值要大于100,不然不当作特征点
    value =  pic[i,j]
    # 得到8个点的坐标信息
    for pppp in range(1):
        point1 = (i - 1, j - 1);
        point2 = (i - 1, j);
        point3 = (i - 1, j + 1);
        point4 = (i, j + 1);
        point5 = (i + 1, j + 1);
        point6 = (i + 1, j);
        point7 = (i + 1, j - 1);
        point8 = (i, j - 1);
    count_big = 0;    count_small = 0
    # 共边的四个点计算
    # if abs(getValue(point2, pic) - value) < 50 : count_big = count_big + 1
    # if abs(getValue(point8, pic) - value) < 50 : count_big = count_big + 1
    # if abs(getValue(point6, pic) - value) < 50 : count_big = count_big + 1
    # if abs(getValue(point4, pic) - value) < 50 : count_big = count_big + 1

    if abs(getValue(point2, pic) - value) < 50 or getValue(point2, pic)>=100 :count_big = count_big + 1
    if abs(getValue(point8, pic) - value) < 50 or getValue(point8, pic)>=100: count_big = count_big + 1
    if abs(getValue(point6, pic) - value) < 50 or getValue(point6, pic)>=100: count_big = count_big + 1
    if abs(getValue(point4, pic) - value )< 50 or getValue(point4, pic)>=100: count_big = count_big + 1
    # 共边4点有两个及以上的点,满足差距很小,小于五十,说明  原始的该点处于线上
    if count_big>=2 :
        return  0

    if (value / getValue(point1,pic)) >2.0  : count_small = count_small + 1
    if (value / getValue(point2,pic)) >2.0 : count_small = count_small + 1
    if (value / getValue(point3, pic)) >2.0: count_small = count_small + 1
    if (value / getValue(point4, pic)) >2.0: count_small = count_small + 1
    if (value / getValue(point5, pic)) >2.0: count_small = count_small + 1
    if (value / getValue(point6, pic)) >2.0: count_small = count_small + 1
    if (value / getValue(point7, pic)) >2.0: count_small = count_small + 1
    if (value / getValue(point8, pic)) >2.0: count_small = count_small + 1
    #周围的8嗝点有5个及以上的数量与该中心点的差距大于80 ,说明该点是断开的点

    if count_small>=5 : return 1
    return 0

def transGRAY2RGB(origin_pic_gray, output_rgb_pic):
    import numpy as np
    import cv2
    file_saveplace = origin_pic_gray

    src = cv2.imread(file_saveplace, 0)
    # print(src)
    src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    # print(src_RGB)
    cv2.imwrite(output_rgb_pic, src_RGB)
    cv2.imshow("rgb", src_RGB)
    # 停留1ms
    cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import cv2
    import numpy as np
    
    # 要检测的图片
    # pic_to_detect =  'thin_origin.png'
    # pic_to_detect =  'thin.png'
    for i in range(0, 19 + 1, 1):
        name1 = 'images//' + '0004_01.pngori_' + str(i) + '.png'
        name2 = 'images//' + '0004_01.pngthin_' + str(i) + '.png'
        the = cv2.imread('the//' + '0004_01the_' + str(0) + '.png')
        the = cv2.imread('the//the256//' + '0004_01the_' + str(i) + '.png', 0)
        pic_to_detect = name2
        transGRAY2RGB(pic_to_detect, pic_to_detect.replace('.png', 'color.png'))
        pic_to_detect_color = pic_to_detect.replace('.png', 'color.png')
        the = cv2.imread('the//the256//' + '0004_01the_' + str(i) + '.png', 0)
        pic_color = cv2.imread(pic_to_detect_color)  # 彩色图保存

        pic = cv2.imread(pic_to_detect, 0)
        print(pic.shape)  # 尺寸是 256* 256

        # labels 记录是否为特征点？是特征点的话该像素位置   取1
        labels = np.zeros(shape=(pic.shape))

        height, width = pic.shape

        # 遍历像素点 (3*3观察)
        for i in range(0 + 1, height - 1):
            for j in range(0 + 1, width - 1):
                # 该点的位置
                point_place__temp = (i, j)
                # 判断是否为特征点 端点？交叉点？(交叉点还没考虑)
                # value = pic[i, j]  # 该点的数值
                labels[i, j] = judgePoint_bili(point_place__temp, pic=pic )
        print(labels)



        # 在原图上绘制点,
        coordinates = []
        for i in range(0 + 1, height - 1):
            for j in range(0 + 1, width - 1):
                if (labels[i, j] == 1): coordinates.append((i, j))
        print(coordinates)
        #保存结果
        pic_todraw = 'outcome_new//dis_method//' + pic_to_detect.replace('.png', '') + 'draw.png'

        drawFeaturePoint(pic_origin=pic_color,
                         coordinates=coordinates,
                         output_savePlace=pic_todraw)



