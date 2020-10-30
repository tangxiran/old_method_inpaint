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


def makedir(dir):
    import os
    dir = dir.strip()
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        return True
    else:
        return False

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

if __name__ == '__main__':
    # 要转换的图片目录是
    origin_gray_filePath = 'F://edge-ours//data_mask//'
    rgb_filePath = 'F://edge-ours//data_mask//rgb_origin//'
    makedir(rgb_filePath)
    file_list = getFileName(origin_gray_filePath)
    for file in file_list[0]:
        #
        origin_filename = origin_gray_filePath + file
        rgb_filename = rgb_filePath +file
        print('old file is '+origin_filename,'\n','rgb file is '+rgb_filename)
        transGRAY2RGB(origin_filename, rgb_filename)