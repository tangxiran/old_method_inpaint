# 返回的是补全结果？
import cv2
import numpy as np
import matplotlib.pyplot as plt

def tv_inpaint(pic_array,mask_array):
    # tv算法
    height,width = pic_array.shape
    pic_copy = pic_array.copy()

    lambda_number = 0.2
    a = 0.5  # 避免分母为0.

    for i in range(0 + 1, height - 1):
        for j in range(0 + 1, width - 1):
            if mask_array[i, j] == 0:
                Un = np.sqrt((pic_array[i, j] - pic_array[i - 1, j]) ** 2 + ((pic_array[i - 1, j - 1] - pic_array[i - 1, j + 1]) / 2) ** 2);
                # Ue = np.sqrt((img(i, j) - img(i, j + 1)) ^ 2 + ((img(i - 1, j + 1) - img(i + 1, j + 1)) / 2) ^ 2)
                Ue = np.sqrt((pic_array[i, j] - pic_array[i, j + 1]) ** 2 + ((pic_array[i - 1, j + 1] - pic_array[i + 1, j + 1]) / 2) ** 2);
                Uw = np.sqrt((pic_array[i, j] - pic_array[i, j - 1]) ** 2 + ((pic_array[i - 1, j - 1] - pic_array[i + 1, j - 1]) / 2) ** 2);
                Us = np.sqrt((pic_array[i, j] - pic_array[i + 1, j]) ** 2 + ((pic_array[i + 1, j - 1] - pic_array[i + 1, j + 1]) / 2) ** 2);

                Wn = 1 / np.sqrt(Un ** 2 + a ** 2);
                We = 1 / np.sqrt(Ue ** 2 + a ** 2);
                Ww = 1 / np.sqrt(Uw ** 2 + a ** 2);
                Ws = 1 / np.sqrt(Us ** 2 + a ** 2);

                Hon = Wn / ((Wn + We + Ww + Ws) + lambda_number);
                Hoe = We / ((Wn + We + Ww + Ws) + lambda_number);
                How = Ww / ((Wn + We + Ww + Ws) + lambda_number);
                Hos = Ws / ((Wn + We + Ww + Ws) + lambda_number);

                Hoo = lambda_number / ((Wn + We + Ww + Ws) + lambda_number);
                value = Hon * pic_array[i - 1, j]+ Hoe * pic_array[i, j + 1]+ How * pic_array[i, j - 1] + Hos * pic_array[i + 1,
                                                                        j] + Hoo * pic_array[i, j];
                pic_copy[i, j] = value;
    # 更新
    pic_array = pic_copy
    return pic_array
def gen_pic_with_mask(mask,origin_pic):
    height,width = origin_pic.shape
    for i in range(height):
        for j in range(width):
            if mask[i,j]==0:
                origin_pic[i,j]=255

    return origin_pic

def makedir(dir):
    import os
    dir = dir.strip()
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        return True
    else:
        return False

if __name__ == '__main__':
    # 要处理的图片
    result = 'resultnew//'; makedir(result)
    pic = 'train2//origin.png'

    # 掩膜，有污损的区域是255白色
    mask = 'train2//newmask.png'
    epochs = 15000
    pic = np.float64(cv2.imread(pic,flags=0))
    mask = cv2.imread(mask,flags=0)
    height ,width = pic.shape
    # 产生图片带有mask
    pic = gen_pic_with_mask(mask=mask,origin_pic=pic)
    cv2.imshow('noise_pic  ', pic)
    cv2.imwrite(filename=result + str('pic_with_mask') + '.png', img=pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    lambda_number =0.2
    a =  0.5 # 避免分母为0.
    for epoch in range(epochs):
        pic_copy = pic.copy()
        pic = tv_inpaint(pic_array=pic_copy,mask_array=mask)
        # 每100次显示一次数据，保存一次数据
        if epoch % 100 ==0:
            cv2.imshow('inpainted_pic'+str(epoch), pic)
            print('epoch,当前的循环次数：',epoch)
            cv2.waitKey(delay=1*1000)
            cv2.destroyAllWindows()
            cv2.imwrite(filename=result+str(epoch)+'.png',img=pic)
            # 再加个阈值处理就够了。但是不适用于大的缺失空白