import cv2
file  = '../fin.png' # 原图
mask = '../origin_mask.png' #  遮盖mask
pic_to_inpainted = cv2.imread(filename=file,flags=0)
mask = cv2.imread(mask,flags=0)

# 下面三个参数
inpainted = cv2.inpaint(pic_to_inpainted,mask,3 , flags=cv2.INPAINT_TELEA)
cv2.imshow('inpainted pic' , inpainted)
cv2.waitKey(0)
cv2.destroyAllWindows()