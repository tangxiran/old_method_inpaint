# 遍历得到某个文件夹下的所有文件名
def getFileName(dirName):
    import os
    fileList= []
    filePath = dirName
    for i, j, k in os.walk(filePath):
        # i是当前路径，j得到文件夹名字，k得到文件名字
        print(i, j, k)
        fileList.append(k)
    return fileList[0] # 返回的是所有的文件的名字列表

def makedir(dir):
    # 新建不存在的文件夹
    import os
    dir = dir.strip()
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        return True
    else:
        return False

if __name__ == '__main__':
    import cv2
    # 要加洞的文件夹名字
    origin_file_dir = r'X://train//thintwofinger//'
    # 要保存在哪个文件夹下（可以是新的文件夹）
    pic_with_mask_you_want_to_save_dir = 'F://new_mask//'
    makedir(pic_with_mask_you_want_to_save_dir)
    # mask的位置
    mask_file  = '../mask.png'
    mask = cv2.imread(mask_file, flags=0)

    file_list =  getFileName(origin_file_dir)
    for pic_file_place in file_list:
        origin_pic_file_savePlace = origin_file_dir + pic_file_place # 你要处理的图片
        pic__to_save_place = pic_with_mask_you_want_to_save_dir + pic_file_place# 处理的图片保存的位置


        pic = cv2.imread(origin_pic_file_savePlace, flags=0)

        height, width = mask.shape
        for i in range(height):
            for j in range(width):
                if mask[i, j] == 255: pic[i, j] = 255
        cv2.imwrite(pic__to_save_place,pic)
        # cv2.imshow('deal ',pic)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()