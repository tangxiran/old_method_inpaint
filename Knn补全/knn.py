# 采取knn最近邻投票修补
import cv2
import  numpy as np
def savenpyasexcel(ndarray,output):
    import pandas as pd
    import  numpy as np
    data_df = pd.DataFrame(ndarray)  # 关键1，将ndarray格式转换为DataFrame
    rows,cols = ndarray.shape
    # 更改表的索引
    data_index = []
    for i in range(rows):
        data_index.append(i)
    data_df.index = data_index
    # 更改表的索引
    data_index = []
    for i in range(cols):
        data_index.append(i)
    data_df.index = data_index
    data_df.columns = data_index

    # 将文件写入excel表格中
    writer = pd.ExcelWriter(output)  # 关键2，创建名称为hhh的excel表格
    data_df.to_excel(writer, 'page_2',
                     float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    writer.save()  # 关键4
    return 1
if __name__ == '__main__':
    file  = '../fin.png'
    mask = '../origin_mask.png'
    pic_to_inpainted = cv2.imread(filename=file,flags=0)
    mask = cv2.imread(filename=mask,flags=0)
    height , width = pic_to_inpainted.shape
    # 要修补的图像，定下flag，数组中要修补的设置为1
    flag_to_inpainted = np.zeros(shape=pic_to_inpainted.shape)
    for i in range(height):
        for j in range(width):
            if mask[i,j]==255 :flag_to_inpainted[i,j] =1
    print(flag_to_inpainted)

