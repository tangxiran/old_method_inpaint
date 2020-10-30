import cv2
import numpy as np

def makedir(dir):
    import os
    dir = dir.strip()
    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
        return True
    else:
        return False

from skimage.feature import canny
img = '0000_00_1.png'# the pic needed to canny deal
img_save = './/canny_deal//'
makedir(img_save)
img_save = img_save+ img
img = cv2.imread(img, flags=0 )



result  = canny(img ).astype(np.float)
result = result * 255
cv2.imwrite(filename=img_save,img=result)
edge = cv2.imread(img_save,0) # result save place

print(edge)
cv2.imshow('origin ', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('res ', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
