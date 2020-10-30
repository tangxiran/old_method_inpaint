# 同一张图片里，寻找相似的5*5的batch块补全图像缺失的部分,按像素补全

# 模板匹配采取误差平方和最小ssd算法
import cv2
import  numpy as np

# 遍历得到某个文件夹下的所有文件名
def getFileName(dirName):
    import os
    fileList= []
    filePath = dirName
    for i, j, k in os.walk(filePath):
        # i是当前路径，j得到文件夹名字，k得到文件名字
        print(i, j, k)
        fileList.append(k)
    return fileList

def makedir(dir):
    # 新建文件夹
    import os
    dir = dir.strip()
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        return True
    else:
        return False

def beleive_assiginment():
    # 增加一个置信度限制
    return  0

def ssd_distance_two_batch(batch1,batch2,pic,flag_to_deal):
    # batch1，batch2是一个列表，里面有25个点
    # pic是图片的array
    # flag是参考的是否需要更新的array
    sum = 0.0
    count = 0
    # batch是如下格式【(x1,y1),(x2,y2)----25个点】
    # batch1代表待填补的区域块
    # batch2代表待匹配的图像上的其他区域快
    for x in range(len(batch1)):
        # print('right')
        tempx1,tempy1 = batch1[x]
        tempx2, tempy2 = batch2[x]
        if flag_to_deal[tempx1,tempy1] == 0 :# 表示该区域本来就存在原始的数值，或是更新后的新数值，可以拿来计算最近邻
            sum = sum + (pic[tempx1,tempy1] - pic[tempx2,tempy2])**2 #ssd距离衡量
            count = count + 1
    batch2_middle_x,batch2_middle_y  =  batch2[len(batch2)//2]
    return  sum *1.0/count ,pic[batch2_middle_x,batch2_middle_y]
    # 返回值是两个batch的距离,以及第二个batch的中心点的数值（可以拿来填充第一个batch）

def get_flag_point(point, flag):
    x,y = point # x,和y超过范围了？？
    return flag[x,y]

def get_importance(little_batch,picture_array, flag_point , believe_importance_ndarray , importance_number_temp ):
    # believe_ndarray 代表置信度矩阵，结合高频信息 取折中
    # 在一个小batch里面，求已知点的区域中，计算方差
    # little_batch 是5*5的小区域，[(x1,y1),(x2,y2)...]
    # flag_point代表先前的flag矩阵，0代表可信值
    count_point_has_value = 0
    sum_point_has_value = 0.0
    for pointx, pointy in little_batch:
        flag_number  = get_flag_point(point=(pointx,pointy) , flag=flag_point)
        # print('到这都是i对的')

        if flag_number==0 :
            # 该点已经有值了,把他加入计算到求importance里面
            sum_point_has_value = sum_point_has_value + picture_array[pointx,pointy]
            count_point_has_value = count_point_has_value + 1
    # 容易得到count_point_value=0 无法当作除数
    importance_get = 0.0

    if count_point_has_value==0:
        # 若是该batch处于内部，不补充该位置，处于边缘的进行补充
        importance_get=0.0

    if count_point_has_value>0:
        # 该batch处于边缘上：计算方差等等操作
        avg_value = sum_point_has_value /count_point_has_value

    for pointx, pointy in little_batch:
        if get_flag_point((pointx,pointy) , flag=flag_point)==0 :
            # 该点已经有值了,把他加入计算到求方差里面
            importance_get  = importance_get +  (avg_value - picture_array[pointx,pointy])**2
    print('#'*15,'importance is ',importance_get ,'#'*15)
    import_point_x , import_point_y  = little_batch[len(little_batch) //2]
    believe_number  = believe_importance_ndarray[import_point_x , import_point_y]
    return  importance_get + believe_number * importance_number_temp


def find_batch(pic_array,flag):
    # 找到哪个batch是亟待修补的
    height, width = pic_array.shape
    # 计算要修补的batch重要程度
    importance  = 0
    # 中心点坐标是
    recordX=0;recordY=0;
    for iii in range(0 + 2, height - 2):
        for jjjj in range(0 + 2, width - 2):
            getFlag = get_flag_point(point=(iii,jjjj) ,flag= flag )
            if getFlag == 1:
                # 该点处于待修补区域,求该点的待修补紧急程度,原则是待修补的batch里面，已知值的点求平均，
                #   然后对每个点做差求绝对值，再求和，其中最大的就是高频区域（类似是方差最大的原则当作高频）
                batch_of_point = get_one_batch( pointXY =  (iii,jjjj) ) # 得到 i,j 为中心的 5*5 小 batch
                if get_importance(little_batch=batch_of_point,
                                  picture_array= pic_array ,
                                  flag_point = flag,
                                  believe_importance_ndarray=beleive_ndarray ,
                                  importance_number_temp=importace_number)  >=  importance:

                    importance = get_importance(little_batch=batch_of_point,  picture_array= pic_array , flag_point = flag
                                                ,believe_importance_ndarray=beleive_ndarray,importance_number_temp= importace_number)
                    batch_record =  batch_of_point
                    recordX=iii;recordY=jjjj;
    return batch_record ,recordX,recordY# 找到了亟需待补全的batch, 和要记录的坐标x,y

def find_point_to_inpaint(pic , flag):
    # 在待修补图像上寻找最需要修补的像素点位置
    # 寻找原则是高频信息
    point_sets=  []
    for i in range(height):
        for j in range(width):
            if flag[i, j] == 1: point_sets.append((i,j))
    # point_sets包含所有需要修补的点


    for x,y in point_sets:
        # 选择一个点，找到周围5*5-1=24个点
        point_side = []
        # 标注24个点是否是已经填充区域

        for i in range(24):
            point_side.append((x - 2, y  - 2));point_side.append((x - 2, y - 1 ));point_side.append((x - 2, y ));point_side.append((x - 2 , y  + 1));point_side.append((x - 2 , y  + 2));
            point_side.append((x - 1 , y  - 2));point_side.append((x - 1 , y - 1 ));point_side.append((x - 1 , y )); point_side.append((x - 1  , y  + 1)); point_side.append((x - 1  , y  + 2));
            point_side.append((x  , y  - 2));point_side.append((x  , y - 1 ));point_side.append((x  , y )); point_side.append((x   , y  + 1));
            point_side.append((x + 1 , y  - 2));point_side.append((x + 1 , y - 1 ));point_side.append((x + 1 , y )); point_side.append((x + 1  , y  + 1)); point_side.append((x + 1  , y  + 2));
            point_side.append((x + 2 , y  - 2));point_side.append((x + 2 , y - 1 ));point_side.append((x + 2 , y )); point_side.append((x + 2  , y  + 1)); point_side.append((x + 2  , y  + 2));
            # 该点是否为已经填充的区域
        flag_of_side = []
        for dotx,doty in point_side:

            if get_flag_point((dotx,doty),flag) == 0: flag_of_side.append( 1 );
            else: flag_of_side.append(0);
        print(flag_of_side) # flagside表示周围24个点，其中哪个点是本来就存在值的，标为1

def get_one_batch(pointXY):
    i,j = pointXY
    temp_batch  = []
    for temp_point in range(1):
        tempPoint = (i - 2, j - 2); temp_batch.append(tempPoint);
        tempPoint = (i - 2, j - 1); temp_batch.append(tempPoint);
        tempPoint = (i - 2, j);     temp_batch.append(tempPoint);
        tempPoint = (i - 2, j+1);     temp_batch.append(tempPoint);
        tempPoint = (i - 2, j+2);   temp_batch.append(tempPoint);
        tempPoint = (i - 1, j - 2);
        temp_batch.append(tempPoint);
        tempPoint = (i - 1, j - 1);
        temp_batch.append(tempPoint);
        tempPoint = (i - 1, j);
        temp_batch.append(tempPoint);
        tempPoint = (i - 1, j + 1);
        temp_batch.append(tempPoint);
        tempPoint = (i - 1, j + 2);
        temp_batch.append(tempPoint);
        tempPoint = (i , j - 2);
        temp_batch.append(tempPoint);
        tempPoint = (i , j - 1);
        temp_batch.append(tempPoint);
        tempPoint = (i , j);
        temp_batch.append(tempPoint);
        tempPoint = (i , j + 1);
        temp_batch.append(tempPoint);
        tempPoint = (i , j + 2);
        temp_batch.append(tempPoint);
        tempPoint = (i + 1, j - 2);
        temp_batch.append(tempPoint);
        tempPoint = (i + 1, j - 1);
        temp_batch.append(tempPoint);
        tempPoint = (i + 1, j);
        temp_batch.append(tempPoint);
        tempPoint = (i + 1, j + 1);
        temp_batch.append(tempPoint);
        tempPoint = (i + 1, j + 2);
        temp_batch.append(tempPoint);
        tempPoint = (i + 2, j - 2);
        temp_batch.append(tempPoint);
        tempPoint = (i + 2, j - 1);
        temp_batch.append(tempPoint);
        tempPoint = (i + 2, j);
        temp_batch.append(tempPoint);
        tempPoint = (i + 2, j + 1);
        temp_batch.append(tempPoint);
        tempPoint = (i + 2, j + 2);
        temp_batch.append(tempPoint);
    # temp_batch 是[(x1,y1) ,(x2,y2)----25个点]

    return temp_batch

def get_all_batch(pic,flag):
    # 一个batch的尺寸是5*5
    all_batch = [] # 符合条件的所有batch，条件就是该batch内不存在未知像素点
    height,width  =pic.shape
    for i in range(0+2,height-2):
        for j in range(0+2,width-2):
            temp_batch = []
            flag_is_batch = 1
            # 对区域进行判断，有未知点的区域都不能算作可以拿来匹配的区域
            for panduan in range(1):
                ################################
                tempPoint = (i-2,j-2);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint ,flag=flag)==1 : flag_is_batch=0
                tempPoint = (i-2,j-1);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint ,flag=flag)==1 : flag_is_batch=0
                tempPoint = (i-2,j );temp_batch.append(tempPoint);
                if get_flag_point(tempPoint ,flag=flag)==1 : flag_is_batch=0
                tempPoint = (i-2,j +1);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint ,flag=flag)==1 : flag_is_batch=0
                tempPoint = (i-2,j+2 );temp_batch.append(tempPoint);
                if get_flag_point(tempPoint ,flag=flag)==1 : flag_is_batch=0
                ################################
                tempPoint = (i - 1, j - 2);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i - 1, j - 1);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i - 1, j);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i - 1, j + 1);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i - 1, j + 2);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0

                ##################
                tempPoint = (i, j - 2);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i, j - 1);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i, j );temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i, j + 1);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i, j + 2);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0

                #########
                tempPoint = (i + 1, j - 2);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i + 1, j - 1);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i + 1, j);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i + 1, j + 1);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i + 1, j + 2);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                ############
                tempPoint = (i + 2, j - 2);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i + 2, j - 1);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i + 2, j);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i + 2, j + 1);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
                tempPoint = (i + 2, j + 2);temp_batch.append(tempPoint);
                if get_flag_point(tempPoint, flag=flag) == 1: flag_is_batch = 0
            if flag_is_batch==1 :
                # 该区域不存在未知点，加到allbatch里面
                all_batch.append(temp_batch)
    # 返回值是列表,每个列表元素是一个[(x1,y1),*****25个点组成的batch]
    # 到这都是对的
    print('still right to here ')
    return  all_batch

if __name__ == '__main__':
    file_dir   = './canny_deal//'
    file_list  = getFileName(file_dir)
    file_list = file_list[0]
    mask = '../origin_mask.png'

    for pic_one in file_list:
        pic_one_place  = file_dir + pic_one
        for importace_number in range(1000000,1000000+1,10000):

            pic_after_inpainted_save_dir = './canny_deal//importanceX'+str(importace_number)+'_'+ pic_one.replace('.png','')+'//'
            mask = '../origin_mask.png'
            makedir(pic_after_inpainted_save_dir)
            pic_to_inpainted_origin = cv2.imread(filename=pic_one_place ,flags=0)
            pic_to_inpainted = pic_to_inpainted_origin.copy()
            # mask是加的遮盖，
            mask = cv2.imread(filename=mask,flags=0)
            height , width = pic_to_inpainted.shape
            # 要修补的图像，增添一个flag数组作为示意图，设置数组中要修补的像素点的位置设置为1
            flag = np.zeros(shape=pic_to_inpainted.shape)
            for ii in range(height):
                for jj in range(width):
                    if mask[ii,jj]==255 :flag[ii,jj] =1
            print(flag)

            # 、增加一个置信度矩阵，设置已有点的位置的置信度为1 ，越往里的修复结果 置信度越小
            # 之后取得高频信息和置信度  相协调的选择要修修补的块位置 ，
            # 加上置信度是为了考虑一点优先修复边缘的想法 有初始值的设置为1 ， 未知的设置为0
            beleive_ndarray = np.zeros(shape=pic_to_inpainted_origin.shape,dtype=float)
            # 置信度矩阵的原则上，越接近中心的置信度越低 ， 范围 0 -1
            middle_point_x  , middle_point_y  = height//2 , width//2
            for ii in range(height):
                for jj in range(width):
                    the_distance_away = ( (ii-middle_point_x)**2+(jj-middle_point_y)**2 )
                    beleive_ndarray[ii,jj]  = the_distance_away
            max_distance = np.max(beleive_ndarray)
            min_distance =np.min(beleive_ndarray)
            # 归一化
            for ii in range(height):
                for jj in range(width):
                    beleive_ndarray[ii,jj]  = 1-((max_distance-beleive_ndarray[ii,jj])/(max_distance-min_distance))
            # max_distance = np.max(beleive_ndarray)
            # min_distance =np.min(beleive_ndarray)
            print( beleive_ndarray )

            print(flag)



            # np.unique(flag,return_counts=True)
            # 统计 1 的个数
            # 当待填补的图片仍有像素未被填充，执行循环操作
            inpainted_iter_number  = 0

            while( (np.sum(flag == 1)) >0 ):

                inpainted_iter_number  = inpainted_iter_number + 1
                # 保存修复后的照片的位置
                pic_after_inpainted_save_place =  pic_after_inpainted_save_dir \
                                                  + 'inpainted'+str(inpainted_iter_number)+'.png'
                #test
                print('test01 is good ')
                print('还有多少像素点待填充: ',np.sum(flag==1)) # 1245个mask点

                # 遍历得到现存的所有的batch，可以拿来修补像素的5*5小batch，每个batch里面的像素点都是已知的
                allbatch  = get_all_batch( pic= pic_to_inpainted , flag = flag)
                # 那个点最需要补充,获取该像素所在的batch，以及flag
                # 沿着flag寻找最迫切需要修补的像素位置,提取其所在的batch，和坐标x，y
                the_batch_to_inpaint, pointx, pointy = find_batch(pic_to_inpainted, flag=flag)
                # getpoint
                # 遍历所有快，得到最小的距离ssd，用该快的中心值填充该像素点
                # 初始化距离
                min_ssd_distance = 100000000 # 设置初始值很大的一个值
                min_ssd_distance  = float("inf")

                for batch_to_detect in allbatch:
                    # batch只是像素的位置数据，（（x1,y1）,(x2,y2)---25个点）
                    # batch1是你找到的要修复的块，batch2是全图中可靠的块，拿来用的
                    dis , value = ssd_distance_two_batch( batch1 = the_batch_to_inpaint ,
                                                          batch2 = batch_to_detect ,
                                                          pic = pic_to_inpainted ,
                                                          flag_to_deal = flag )

                    # 得到中心点的值，得到dis距离值
                    if min_ssd_distance>dis:
                        # 找到一个更相似的的batch，更新
                        the_value_to_use  =value
                        min_ssd_distance =dis
                        print('所有的小块里面 ，min ssd is ',min_ssd_distance)
                # 用找到的数值填充该像素的值
                pic_to_inpainted[pointx , pointy ] = the_value_to_use

                # 更新flag数组，少了一个待填充的像素值 ，将该位置已赋值的flag设置为0
                flag[pointx , pointy] = 0
                print('该块的最小误差是',min_ssd_distance,'填充的值是',the_value_to_use)
                cv2.imwrite(pic_after_inpainted_save_place,pic_to_inpainted)
                # cv2.imshow('No'+str(inpainted_iter_number),pic_to_inpainted)
                # cv2.waitKey(1)
                # cv2.destroyAllWindows()
            print(pic_to_inpainted) # 最终结果是





