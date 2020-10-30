def biggerThan40(point,pic):
    x,y = point
    if pic[x,y]>40:
        return 1
    else:return 0
def twoArrayToOne(label_new_temp,label_old_temp ):
    # 对应位置相乘,每个元素
    a,b = label_new_temp.shape
    labels_temp = np.zeros(shape=(a,b))
    for i in range(0, a):
        for j in range(0,b):
                labels_temp[i, j] = label_new_temp[i,j] * label_old_temp[i,j]
    return labels_temp
def transGRAY2RGB(origin_pic_gray , output_rgb_pic):
    import numpy as np
    import cv2
    file_saveplace = origin_pic_gray

    src = cv2.imread(file_saveplace, 0)
    # print(src)
    src_RGB = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    # print(src_RGB)
    cv2.imwrite(output_rgb_pic,src_RGB)
    cv2.imshow("rgb",src_RGB)
    # 停留1ms
    cv2.waitKey(1)
    cv2.destroyAllWindows()

def blockFlag(point1,point2,point3 , pic):
    if biggerThan40(point2,pic)==0 and biggerThan40(point1,pic)==0 and biggerThan40(point3,pic)==0 :
        return 1
    # count_big = 0
    # count_small = 0

    return 0

def judegePointNew(point_temp, pic ,the):
    # # 判断是否为特征点 ,
    i,j = point_temp
    # # 判断要求： 周围8个点 ，有一个点的差距小于50，且有5个点的差距大于80
    if not biggerThan40(point_temp, pic) : return 0
    # 得到8个点的坐标信息

        # for pppp in range(1):
        #     point1 = (i,j-1);point2 = (i-1,j-1);point3 = (i-1,j);point4 = (i-1,j+1);
        #     point5 = (i,j+1);point6 = (i+1,j+1);point7 = (i+1,j);point8 = (i+1,j-1);
        #     128,
        #     2,3,4
        #     4,5,6
        #     6,7,8四块

    for ppp in  range(1):
        point1 = (i - 1, j - 1);
        point2 = (i - 1, j);
        point3 = (i - 1, j + 1);
        point4 = (i, j + 1);
        point5 = (i + 1, j + 1);
        point6 = (i + 1, j);
        point7 = (i + 1, j - 1);
        point8 = (i, j - 1);

    count = 0 # 计算flag个数

    leftup = 0;leftdown=0;rightdown=0;rightup=0;
    if blockFlag(point1,point2,point8,pic):count+=1;leftup = 1 ;
    if blockFlag(point6,point7,point8,pic):count+=1;leftdown=1;
    if blockFlag(point4,point5,point6,pic):count+=1 ;rightdown = 1;
    if blockFlag(point2,point3,point4,pic):count+=1; rightup = 1 ;

    if count==2 :return 1

    if count == 1 :
        if leftup==1 :
            if not  biggerThan40( point7 ,pic): return 1
        if leftdown==1 :
            if not  biggerThan40( point1 ,pic): return 1
        if rightup == 1:
            if not biggerThan40(point5,pic): return 1
        if rightdown == 1:
            if not biggerThan40(point3,pic): return 1
    return  0

def judgePoint_useful(point_temp, pic ,the):
    # # 判断是否为特征点 ,
    i,j = point_temp
    # # 判断要求： 周围8个点 ，有一个点的差距小于50，且有5个点的差距大于80
    if not biggerThan40(point_temp, pic) : return 0
    # 得到8个点的坐标信息


    temp_angle = the[i//4,j//4]
    if temp_angle < 255//2:# >90度

        for pppp in range(1):
            # 1,2,8,
            # 2,3,4
            # 4,5,6
            # 6,7,8 四块
            point1 = (i - 1, j - 1);
            point2 = (i - 1, j);
            point3 = (i - 1, j + 1);
            point4 = (i, j + 1);
            point5 = (i + 1, j + 1);
            point6 = (i + 1, j);
            point7 = (i + 1, j - 1);
            point8 = (i, j - 1);

        # if temp < int(255/2): # 有问题!!!
        #     for pppp in range(1):
        #         # 1,2,8,
        #         # 2,3,4
        #         # 4,5,6
        #         # 6,7,8 四块
        #         point1 = (i-1,j-1);point2 = (i-1,j);point3 = (i-1,j+1);point4 = (i,j+1);
        #         point5 = (i+1,j+1);point6 = (i+1,j);point7 = (i+1,j-1);point8 = (i,j-1);
        # else:
        #     for ppp in range(1):
        #         point1 = (j - 1, i - 1);
        #         point2 = (j, i - 1);
        #         point3 = (j + 1, i - 1);
        #         point4 = (j + 1, i);
        #         point5 = (j + 1, i + 1);
        #         point6 = (j, i + 1);
        #         point7 = (j - 1, i + 1);
        #         point8 = (j - 1, i);
        #         # 1,2,8,
        #         # 2,3,4
        #         # 4,5,6
        #         # 6,7,8 四块
        count = 0 # 计算flag个数
        # if  biggerThan40(point_temp,pic):
        #     if not biggerThan40(point3,pic) :
        #         if( (biggerThan40((point3[0]-1,point3[1]-1),pic) )==0 )and (biggerThan40((point3[0],point3[1]-1),pic) ==0) and (biggerThan40((point3[0]-1,point3[1]),pic) ==0)  :
        #             count = count+1
        #
        leftup = 0;leftdown=0;rightdown=0;rightup=0;
        if blockFlag(point1,point2,point8,pic):count+=1;leftup = 1 ;
        if blockFlag(point6,point7,point8,pic):count+=1;leftdown=1;
        if blockFlag(point5,point6,point4,pic):count+=1 ;rightdown = 1;
        if blockFlag(point4,point2,point3,pic):count+=1; rightup = 1 ;
        if count==0 : return  0
        if count==2 :return 1

        if count == 1 :
            if leftup==1 :
                if not  biggerThan40( point7 ,pic): return 1
            if leftdown==1 :
                if not  biggerThan40( point1 ,pic): return 1
            if rightup == 1:
                if not biggerThan40(point5,pic): return 1
            if rightdown == 1:
                if not biggerThan40(point3,pic): return 1
        return  0
    if temp_angle >= 255 // 2:  # >90度
        for pppp in range(1):
            # 1,2,8,
            # 2,3,4
            # 4,5,6
            # 6,7,8 四块
            point1 = (i - 1, j - 1);
            point2 = (i - 1, j);
            point3 = (i - 1, j + 1);
            point4 = (i, j + 1);
            point5 = (i + 1, j + 1);
            point6 = (i + 1, j);
            point7 = (i + 1, j - 1);
            point8 = (i, j - 1);

        count = 0  # 计算flag个数
        # if  biggerThan40(point_temp,pic):
        #     if not biggerThan40(point3,pic) :
        #         if( (biggerThan40((point3[0]-1,point3[1]-1),pic) )==0 )and (biggerThan40((point3[0],point3[1]-1),pic) ==0) and (biggerThan40((point3[0]-1,point3[1]),pic) ==0)  :
        #             count = count+1
        #
        leftup = 0;
        leftdown = 0;
        rightdown = 0;
        rightup = 0;
        if blockFlag(point1, point2, point8, pic): count += 1;leftup = 1;
        if blockFlag(point6, point7, point8, pic): count += 1;leftdown = 1;
        if blockFlag(point5, point6, point4, pic): count += 1;rightdown = 1;
        if blockFlag(point4, point2, point3, pic): count += 1; rightup = 1;
        if count == 0: return 0
        if count == 2: return 1

        if count == 1:

            if rightup == 1:
                return 1
            if rightdown == 1:
                return 1
        return 0

def judgePoint_good(point_temp, pic ,the):
    # # 判断是否为特征点 ,
    i,j = point_temp
    # # 判断要求： 周围8个点 ，有一个点的差距小于50，且有5个点的差距大于80
    if not biggerThan40(point_temp, pic) : return 0
    # 得到8个点的坐标信息


    temp_angle = the[i//4,j//4]
    if temp_angle < 255//2:# >90度

        for pppp in range(1):
            # 1,2,8,
            # 2,3,4
            # 4,5,6
            # 6,7,8 四块
            point1 = (i - 1, j - 1);
            point2 = (i - 1, j);
            point3 = (i - 1, j + 1);
            point4 = (i, j + 1);
            point5 = (i + 1, j + 1);
            point6 = (i + 1, j);
            point7 = (i + 1, j - 1);
            point8 = (i, j - 1);

        # if temp < int(255/2): # 有问题!!!
        #     for pppp in range(1):
        #         # 1,2,8,
        #         # 2,3,4
        #         # 4,5,6
        #         # 6,7,8 四块
        #         point1 = (i-1,j-1);point2 = (i-1,j);point3 = (i-1,j+1);point4 = (i,j+1);
        #         point5 = (i+1,j+1);point6 = (i+1,j);point7 = (i+1,j-1);point8 = (i,j-1);
        # else:
        #     for ppp in range(1):
        #         point1 = (j - 1, i - 1);
        #         point2 = (j, i - 1);
        #         point3 = (j + 1, i - 1);
        #         point4 = (j + 1, i);
        #         point5 = (j + 1, i + 1);
        #         point6 = (j, i + 1);
        #         point7 = (j - 1, i + 1);
        #         point8 = (j - 1, i);
        #         # 1,2,8,
        #         # 2,3,4
        #         # 4,5,6
        #         # 6,7,8 四块
        count = 0 # 计算flag个数
        # if  biggerThan40(point_temp,pic):
        #     if not biggerThan40(point3,pic) :
        #         if( (biggerThan40((point3[0]-1,point3[1]-1),pic) )==0 )and (biggerThan40((point3[0],point3[1]-1),pic) ==0) and (biggerThan40((point3[0]-1,point3[1]),pic) ==0)  :
        #             count = count+1
        #
        leftup = 0;leftdown=0;rightdown=0;rightup=0;
        if blockFlag(point1,point2,point8,pic):count+=1;leftup = 1 ;
        if blockFlag(point6,point7,point8,pic):count+=1;leftdown=1;
        if blockFlag(point5,point6,point4,pic):count+=1 ;rightdown = 1;
        if blockFlag(point4,point2,point3,pic):count+=1; rightup = 1 ;
        if count==0 : return  0
        if count==2 :return 1

        if count == 1 :
            if leftup==1 :
                if not  biggerThan40( point7 ,pic): return 1
            if leftdown==1 :
                if not  biggerThan40( point1 ,pic): return 1
            if rightup == 1:
                if not biggerThan40(point5,pic): return 1
            if rightdown == 1:
                if not biggerThan40(point3,pic): return 1
        return  0
    if temp_angle >= 255 // 2:  # >90度
        for pppp in range(1):
            # 1,2,8,
            # 2,3,4
            # 4,5,6
            # 6,7,8 四块
            point1 = (i - 1, j - 1);
            point2 = (i - 1, j);
            point3 = (i - 1, j + 1);
            point4 = (i, j + 1);
            point5 = (i + 1, j + 1);
            point6 = (i + 1, j);
            point7 = (i + 1, j - 1);
            point8 = (i, j - 1);

        count = 0  # 计算flag个数
        # if  biggerThan40(point_temp,pic):
        #     if not biggerThan40(point3,pic) :
        #         if( (biggerThan40((point3[0]-1,point3[1]-1),pic) )==0 )and (biggerThan40((point3[0],point3[1]-1),pic) ==0) and (biggerThan40((point3[0]-1,point3[1]),pic) ==0)  :
        #             count = count+1
        #
        leftup = 0;
        leftdown = 0;
        rightdown = 0;
        rightup = 0;
        if blockFlag(point1, point2, point8, pic): count += 1;leftup = 1;
        if blockFlag(point6, point7, point8, pic): count += 1;leftdown = 1;
        if blockFlag(point5, point6, point4, pic): count += 1;rightdown = 1;
        if blockFlag(point4, point2, point3, pic): count += 1; rightup = 1;
        if count == 0: return 0
        if count == 2: return 1

        if count == 1:

            if rightup == 1:
                return 1
            if rightdown == 1:
                return 1
        return 0


def judegePointOld(point_temp, pic ,the):
    # 早期的方法一
    # # 判断是否为特征点 ,
    i,j = point_temp

    if not biggerThan40(point_temp, pic) : return 0
    # 得到8个点的坐标信息
    # 123, 345,567,781四块

    for ppp in  range(1):
        point1 = (i , j -1);
        point2 = (i - 1, j-1);
        point3 = (i-1 , j );
        point4 = (i-1, j + 1);
        point5 = (i , j +1);
        point6 = (i + 1, j+1);
        point7 = (i +1, j );
        point8 = (i+1, j - 1);

    count = 0 # 计算flag个数
    # if  biggerThan40(point_temp,pic):
    #     if not biggerThan40(point3,pic) :
    #         if( (biggerThan40((point3[0]-1,point3[1]-1),pic) )==0 )and (biggerThan40((point3[0],point3[1]-1),pic) ==0) and (biggerThan40((point3[0]-1,point3[1]),pic) ==0)  :
    #             count = count+1
    #
    leftup = 0;leftdown=0;rightdown=0;rightup=0;
    if blockFlag(point1,point2,point3,pic):count+=1;leftup = 1 ;
    if blockFlag(point3,point4,point5,pic):count+=1;leftdown=1;
    if blockFlag(point5,point6,point7,pic):count+=1 ;rightdown = 1;
    if blockFlag(point7,point8,point1,pic):count+=1; rightup = 1 ;

    if count==2 :return 1

    if count == 1 :
        if leftup==1 :
            if not  biggerThan40( point4 ,pic): return 1
        if leftdown==1 :
            if not  biggerThan40( point2 ,pic): return 1
        if rightup == 1:
            if not biggerThan40(point6,pic): return 1
        if rightdown == 1:
            if not biggerThan40(point8,pic): return 1
    return  0

def judgePoint(point_temp, pic ,the): 
    # 新的判断方法
    # # 判断是否为特征点 ,
    i,j = point_temp
    
    if not biggerThan40(point_temp, pic) : return 0
    # 得到8个点的坐标信息

    for pppp in range(1):
        # 1,2,8,
        # 2,3,4
        # 4,5,6
        # 6,7,8 四块
        point1 = (i - 1, j - 1);
        point2 = (i - 1, j);
        point3 = (i - 1, j + 1);
        point4 = (i, j + 1);
        point5 = (i + 1, j + 1);
        point6 = (i + 1, j);
        point7 = (i + 1, j - 1);
        point8 = (i, j - 1);

    # if temp < int(255/2): # 有问题!!!
    #     for pppp in range(1):
    #         # 1,2,8,
    #         # 2,3,4
    #         # 4,5,6
    #         # 6,7,8 四块
    #         point1 = (i-1,j-1);point2 = (i-1,j);point3 = (i-1,j+1);point4 = (i,j+1);
    #         point5 = (i+1,j+1);point6 = (i+1,j);point7 = (i+1,j-1);point8 = (i,j-1);
    # else:
    #     for ppp in range(1):
    #         point1 = (j - 1, i - 1);
    #         point2 = (j, i - 1);
    #         point3 = (j + 1, i - 1);
    #         point4 = (j + 1, i);
    #         point5 = (j + 1, i + 1);
    #         point6 = (j, i + 1);
    #         point7 = (j - 1, i + 1);
    #         point8 = (j - 1, i);
    #         # 1,2,8,
    #         # 2,3,4
    #         # 4,5,6
    #         # 6,7,8 四块
    count = 0 # 计算flag个数
    # if  biggerThan40(point_temp,pic):
    #     if not biggerThan40(point3,pic) :
    #         if( (biggerThan40((point3[0]-1,point3[1]-1),pic) )==0 )and (biggerThan40((point3[0],point3[1]-1),pic) ==0) and (biggerThan40((point3[0]-1,point3[1]),pic) ==0)  :
    #             count = count+1
    #
    leftup = 0;leftdown=0;rightdown=0;rightup=0;
    if blockFlag(point1,point2,point8,pic):count+=1;leftup = 1 ;
    if blockFlag(point6,point7,point8,pic):count+=1;leftdown=1;
    if blockFlag(point5,point6,point4,pic):count+=1 ;rightdown = 1;
    if blockFlag(point4,point2,point3,pic):count+=1; rightup = 1 ;
    if count==0 : return  0
    if count==2 :return 1

    if count == 1 :
        if leftup==1 :
            if not  biggerThan40( point7 ,pic): return 1
        if leftdown==1 :
            if not  biggerThan40( point1 ,pic): return 1
        if rightup == 1:
            if not biggerThan40(point5,pic): return 1
        if rightdown == 1:
            if not biggerThan40(point3,pic): return 1
    return  0



if __name__ == '__main__':

    import cv2

    import numpy as np

    # 要检测的图片
    # pic_to_detect =  'thin_origin.png'
    # pic_to_detect =  'thin.png'
    for i in range(0,19+1,1):
        the = cv2.imread('the//' + '0004_01the_' + str(0) + '.png')
        the = cv2.imread('the//the256//' + '0004_01the_' + str(0) + '.png', 0)
        name1 ='images//'+'0004_01.pngori_'+ str(i)+'.png'
        name2 = 'images//'+'0004_01.pngthin_'+ str(i)+'.png'
        pic_to_detect =  name2
        transGRAY2RGB(pic_to_detect , pic_to_detect.replace('.png','color.png'))
        pic_to_detect_color = pic_to_detect.replace('.png','color.png')
        pic_color = cv2.imread(pic_to_detect_color) #彩色图保存

        pic = cv2.imread(pic_to_detect,0)
        print(pic.shape ) # 尺寸是 256* 256

        # labels 记录是否为特征点？是特征点的话该像素位置   取1
        label_new= np.zeros(shape=(pic.shape))
        label_old  = np.zeros(shape=(pic.shape))

        height  , width = pic.shape
        # 早期方法的图片:
        for old in range(1):
            label_new = np.zeros(shape=(pic.shape))
            label_old = np.zeros(shape=(pic.shape))
            labels03 = np.zeros(shape=(pic.shape))
            labels_the = np.zeros(shape=(pic.shape))
            labels_good= np.zeros(shape=(pic.shape))
            labels = np.zeros(shape=(pic.shape))
            # 遍历像素点 (3*3观察)
            for i in range(0 + 1, height - 1):
                for j in range(0 + 1, width - 1):
                    # 该点的位置
                    point_place__temp = (i, j)
                    # 判断是否为特征点 端点？交叉点？(交叉点还没考虑)
                    value = pic[i, j]  # 该点的数值
                    label_new[i, j] = judegePointNew(point_place__temp, pic=pic, the=the)
                    labels_the[i, j] = judgePoint_useful(point_place__temp, pic=pic, the=the)
                    label_old[i, j] = judegePointOld(point_place__temp, pic=pic, the=the)
                    labels03[i, j] = judgePoint(point_place__temp, pic=pic, the=the)
                    labels_good[i,j] = judgePoint_good(point_place__temp,pic=pic,the=the)
            print(label_new)
            print(label_old)
            labels = label_old
            # 在原图上绘制点,
            coordinates = []
            for i in range(0 + 1, height - 1):
                for j in range(0 + 1, width - 1):
                    if (labels[i, j] == 1): coordinates.append((i, j))
            print(coordinates)
            # 点的尺寸
            point_size = 1
            point_color = (0, 0, 255)  # BGR
            thickness = 0  # 0 、4、8

            # 函数参数含义如下：
            #
            # img：要画的圆所在的矩形或图像
            # center：圆心坐标，如(100, 100)
            # radius：半径，如
            # 10
            # color：圆边框颜色，如(0, 0, 255)
            # 红色，BGR
            # thickness：正值表示圆边框宽度.负值表示画一个填充圆形
            # lineType：圆边框线型，可为
            # 0，4，8
            # shift：圆心坐标和半径的小数点位数

            for coor in coordinates:
                print(coor)
                # 横纵坐标的位置?
                cv2.circle(pic_color, (int(coor[1]), int(coor[0])), point_size, point_color, thickness)
            pic_todraw = 'outcome//old//' + pic_to_detect.replace('.png', '') + 'draw.png'
            cv2.imwrite(pic_todraw, pic_color, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.waitKey(1)
            cv2.destroyAllWindows()

            cv2.imshow('origin ', pic)
            cv2.waitKey(1)
            cv2.destroyAllWindows()

        # 现在的方法
        for new in range(1):
            label_new = np.zeros(shape=(pic.shape))
            label_old = np.zeros(shape=(pic.shape))
            labels_the = np.zeros(shape=(pic.shape))
            labels_good = np.zeros(shape=(pic.shape))
            labels03 = np.zeros(shape=(pic.shape))
            labels  = np.zeros(shape=(pic.shape))
            for i in range(0+1,height-1):
                for j in range(0+1,width-1):
                    # 该点的位置
                    point_place__temp = (i,j)
                    # 判断是否为特征点 端点？交叉点？(交叉点还没考虑)
                    value = pic[i,j] # 该点的数值
                    label_new[i,j] = judegePointNew(point_place__temp,pic=pic,the=the)
                    labels_the[i,j] = judgePoint_useful(point_place__temp,pic=pic,the=the)
                    label_old[i,j] = judegePointOld(point_place__temp,pic=pic,the=the)
                    labels_good[i,j] = judgePoint_good(point_place__temp,pic=pic,the=the)

            print(label_new)
            print(label_old)
            np.save('npy//label_old.npy',label_old)
            labels_good = np.load('npy//newlabel.npy')
            labels = labels_good

            # 在原图上绘制点,
            coordinates = []
            for i in range(0+1,height-1):
                for j in range(0+1,width-1):
                    if(labels[i,j]==1):coordinates.append((i,j))
            print(coordinates)
            # 点的尺寸
            point_size = 1
            point_color = (0, 0, 255)  # BGR
            thickness = 0  # 0 、4、8

            # 函数参数含义如下：
            #
            # img：要画的圆所在的矩形或图像
            # center：圆心坐标，如(100, 100)
            # radius：半径，如
            # 10
            # color：圆边框颜色，如(0, 0, 255)
            # 红色，BGR
            # thickness：正值表示圆边框宽度.负值表示画一个填充圆形
            # lineType：圆边框线型，可为
            # 0，4，8
            # shift：圆心坐标和半径的小数点位数

            for coor in coordinates:
                print(coor)
                # 横纵坐标的位置?
                cv2.circle(pic_color, (int(coor[1]), int(coor[0])), point_size, point_color, thickness)
            pic_todraw = 'outcome//new//'+ pic_to_detect.replace('.png','') +'draw.png'
            cv2.imwrite(pic_todraw , pic_color, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.waitKey(1)
            cv2.destroyAllWindows()



            cv2.imshow('origin ', pic)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        # 合体的方法
        for all in range(1):
            label_new = np.zeros(shape=(pic.shape))
            label_old = np.zeros(shape=(pic.shape))
            labels_good = np.zeros(shape=(pic.shape))
            labels03 = np.zeros(shape=(pic.shape))
            labels = np.zeros(shape=(pic.shape))
            for i in range(0 + 1, height - 1):
                for j in range(0 + 1, width - 1):
                    # 该点的位置
                    point_place__temp = (i, j)
                    # 判断是否为特征点 端点？交叉点？(交叉点还没考虑)
                    value = pic[i, j]  # 该点的数值
                    label_new[i, j] = judegePointNew(point_place__temp, pic=pic, the=the)
                    labels_the[i,j] = judgePoint_useful(point_place__temp,pic=pic,the=the)
                    labels_good[i,j] = judgePoint_good(point_place__temp,pic=pic,the=the)

                    label_old[i, j] = judegePointOld(point_place__temp, pic=pic, the=the)

            print(label_new)
            print(label_old)
            labels_good = np.load('npy//newlabel.npy')
            labels = twoArrayToOne(label_old, labels_good)

            # 在原图上绘制点,
            coordinates = []
            for i in range(0 + 1, height - 1):
                for j in range(0 + 1, width - 1):
                    if (labels[i, j] == 1): coordinates.append((i, j))
            print(coordinates)
            # 点的尺寸
            point_size = 1
            point_color = (0, 0, 255)  # BGR
            thickness = 0  # 0 、4、8

            # 函数参数含义如下：
            #
            # img：要画的圆所在的矩形或图像
            # center：圆心坐标，如(100, 100)
            # radius：半径，如
            # 10
            # color：圆边框颜色，如(0, 0, 255)
            # 红色，BGR
            # thickness：正值表示圆边框宽度.负值表示画一个填充圆形
            # lineType：圆边框线型，可为
            # 0，4，8
            # shift：圆心坐标和半径的小数点位数

            for coor in coordinates:
                print(coor)
                # 横纵坐标的位置?
                cv2.circle(pic_color, (int(coor[1]), int(coor[0])), point_size, point_color, thickness)
            pic_todraw = 'outcome//all//' + pic_to_detect.replace('.png', '') + 'draw.png'
            cv2.imwrite(pic_todraw, pic_color, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.waitKey(1)
            cv2.destroyAllWindows()

            cv2.imshow('origin ', pic)
            cv2.waitKey(1)
            cv2.destroyAllWindows()


        # 新的的方法 , 在
        for new in range(1):
            label_new = np.zeros(shape=(pic.shape))
            label_old = np.zeros(shape=(pic.shape))
            labels03 = np.zeros(shape=(pic.shape))
            labels = np.zeros(shape=(pic.shape))
            for i in range(0 + 1, height - 1):
                for j in range(0 + 1, width - 1):
                    # 该点的位置
                    point_place__temp = (i, j)
                    # 判断是否为特征点 端点？交叉点？(交叉点还没考虑)
                    value = pic[i, j]  # 该点的数值
                    label_new[i, j] = judegePointNew(point_place__temp, pic=pic, the=the)
                    labels03[i, j] = judgePoint(point_place__temp ,  pic=pic,the=the)
                    label_old[i, j] = judegePointOld(point_place__temp, pic=pic, the=the)

            print(label_new)
            print(label_old)
            labels = labels03

            # 在原图上绘制点,
            coordinates = []
            for i in range(0 + 1, height - 1):
                for j in range(0 + 1, width - 1):
                    if (labels[i, j] == 1): coordinates.append((i, j))
            print(coordinates)
            # 点的尺寸
            point_size = 1
            point_color = (0, 0, 255)  # BGR
            thickness = 0  # 0 、4、8

            # 函数参数含义如下：
            #
            # img：要画的圆所在的矩形或图像
            # center：圆心坐标，如(100, 100)
            # radius：半径，如
            # 10
            # color：圆边框颜色，如(0, 0, 255)
            # 红色，BGR
            # thickness：正值表示圆边框宽度.负值表示画一个填充圆形
            # lineType：圆边框线型，可为
            # 0，4，8
            # shift：圆心坐标和半径的小数点位数

            for coor in coordinates:
                print(coor)
                # 横纵坐标的位置?
                cv2.circle(pic_color, (int(coor[1]), int(coor[0])), point_size, point_color, thickness)
            pic_todraw = 'outcome//new_Right//' + pic_to_detect.replace('.png', '') + 'draw.png'
            cv2.imwrite(pic_todraw, pic_color, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.waitKey(1)
            cv2.destroyAllWindows()

            cv2.imshow('origin ', pic)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
