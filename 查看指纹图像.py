import cv2
# 保存numpy作为excel
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
# originpic = 'origin_mask.png'
originpic = 'fin.png'
# originpic = 'origin.png'
pic = cv2.imread(originpic, 0)

pic_copy  = pic.copy()
import numpy as np
import pandas as pd

# 准备数据
data = pic_copy


savenpyasexcel(pic_copy , 'excel//fin.xlsx')
