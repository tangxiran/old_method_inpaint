import cv2
import numpy as np

from skimage.feature import canny
img = 'mask02.png'# the pic needed to canny deal
img = cv2.imread(img, flags=0 )



result  = canny(img ).astype(np.float)
result =result *255
cv2.imwrite(filename='new.png',img=result)
edge = cv2.imread('edge_see.png',0) # result save place

print(edge)
# cv2.imshow('origin ', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('res ', result)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
