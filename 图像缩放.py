import cv2  
# 读入原图片
origin_pic = '0004_01.pngthe_0.png'
img = cv2.imread('the//'+origin_pic,0)

# 打印出图片尺寸
print(img.shape)
# 将图片高和宽分别赋值给x，y
x, y = img.shape[0:2]

# 显示原图
cv2.imshow('OriginalPicture', img)

# 缩放到原来的二分之一，输出尺寸格式为（宽，高）
img_test1 = cv2.resize(img, (int(y *8), int(x*8)),interpolation=cv2.INTER_NEAREST)
cv2.imwrite('the//the256//'+origin_pic.replace('.png','') + '.png' ,img_test1)
cv2.imshow('resize0', img_test1)
cv2.waitKey(1)
cv2.destroyAllWindows()
# # 最近邻插值法缩放
# # 缩放到原来的四分之一
# img_test2 = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
# cv2.imshow('resize1', img_test2)
# cv2.waitKey()
# cv2.destroyAllWindows()