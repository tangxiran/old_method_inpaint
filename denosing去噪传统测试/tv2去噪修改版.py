import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob, os
# import xlwt
# import tensorflow.compat.v1 as tf
# from scipy.signal import convolve2d


def TV(number, m_imgData, iter, dt, epsilon, lamb):
    NX = m_imgData.shape[0]
    NY = m_imgData.shape[1]
    ep2 = epsilon * epsilon
    I_t = np.array(np.zeros((NX, NY)))
    I_tmp = np.array(np.ones((NX, NY)))  # 用来迭代的噪声图
    I_t = m_imgData.astype(np.float64)
    I_tmp = m_imgData.astype(np.float64)
    data = []
    for t in range(0, iter):
        for i in range(0,NX):  # 一次迭代
            for j in range(0, NY):
                iUp = i - 1
                iDown = i + 1
                jLeft = j - 1
                jRight = j + 1  # 边界处理
                if i == 0:
                    iUp = i
                if NY - 1 == i:
                    iDown = i
                if j == 0:
                    jLeft = j
                if NX - 1 == j:
                    jRight = j
                tmp_x = (I_t[i][jRight] - I_t[i][jLeft]) / 2.0
                tmp_y = (I_t[iDown][j] - I_t[iUp][j]) / 2.0
                tmp_xx = I_t[i][jRight] + I_t[i][jLeft] - 2 * I_t[i][j]
                tmp_yy = I_t[iDown][j] + I_t[iUp][j] - 2 * I_t[i][j]
                tmp_xy = (I_t[iDown][jRight] + I_t[iUp][jLeft] - I_t[iUp][jRight] - I_t[iDown][jLeft]) / 4.0
                tmp_num = tmp_yy * (tmp_x * tmp_x + ep2) + tmp_xx * (tmp_y * tmp_y + ep2) - 2 * tmp_x * tmp_y * tmp_xy
                tmp_den = math.pow(tmp_x * tmp_x + tmp_y * tmp_y + ep2, 1.5)
                I_tmp[i][j] += dt * (tmp_num / tmp_den + (0.5 + lamb[i][j]) * (m_imgData[i][j] - I_t[i][j]))

        for i in range(0, NX):
            for j in range(0, NY):
                I_t[i][j] = I_tmp[i][j]  # 迭代结束
        loss = ((I_t - simage) ** 2).mean()
        if (t % 10 == 0):
            print(loss)
            data.append(loss)
    data = np.array(data)
    return I_t  # 返回去噪图


def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))
    h,w = temp_image.shape

    noise = np.random.randn(h, w) * noise_sigma
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    return noisy_image

#
# def save_result(result, path):  # 保存结果
#     path = path if path.find('.') != -1 else path + '.png'
#     ext = os.path.splitext(path)[-1]
#     if ext in ('.txt', '.dlm'):
#         np.savetxt(path, result, fmt='%2.4f')
#     else:
#         imsave(path, np.clip(result, 0, 1))


def cal_psnr(im1, im2):
    mse = ((im1 - im2) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def save_images(filepath, denoise_image, noisy_image, clean_image):  # 保存图片
    denoise_image = np.squeeze(denoise_image)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = denoise_image
    else:
        cat_image = np.concatenate([clean_image, noisy_image, denoise_image], axis=1)
    cv2.imwrite(filepath, cat_image)


def datagenerator(data_dir):  # 生成图片集
    file_list = glob.glob(data_dir + '/*.png') + glob.glob(data_dir + '/*.png')   # get name list of all .png files
    data = []
    print(file_list)
    for i in range(len(file_list)):
        img = cv2.imread(file_list[i], 0)
        data.append(np.array(img))
    return data


def load_images(filelist):
    file_list =list( glob.glob(data_dir + '/*.png')) +  list( glob.glob(data_dir + '/*.jpg')) # get name list of all .png files
    data = []
    print(file_list)
    for i in range(len(file_list)):
        file_temp = file_list[i]
        im = cv2.imread(filename=file_temp , flags=0) # to gray picture
        data.append(np.array(im))
    return data

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
    noise_sigma = 25
    data_dir = '../tv算法去噪？/train2'
    result = 'result'
    makedir('../tv算法去噪？/result//')
    data = load_images(data_dir)
    psnr0 = []
    psnr1 = []
    for i in range(10):  # 对图像集进行处理,10个图片？？
        str1 = str(i)
        image = add_gaussian_noise(data[i], noise_sigma=noise_sigma)
        simage = data[i].astype(np.float64)  # 原图像
        NX,NY = (data[i]).shape
        lamb = np.array(np.zeros((NX, NY)))
        for t in range(NX):
            for j in range(NY):
                lamb[t][j] = 0
        Img = TV(i, image, 300, 0.1, 1, lamb)
        psnr = cal_psnr(Img, simage)

        path = os.path.join(result, str1 + '.png')  # 保存图像集的路径
        save_images(path, Img, image, simage);  # 保存结果
        # cv2.imwrite(path,Img)
        print("image :", i, "psnr : ", psnr)