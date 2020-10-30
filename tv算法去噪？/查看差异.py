import cv2
import numpy as np
old  = 'result//origin_mask.png'
new =  'result//8200.png'
old = cv2.imread(filename=old,flags=0)
new  = cv2.imread(filename=new,flags=0)
dis = old-new
print(dis)

# a = np.array([[1,2,356],[5241,22,23]]
#              )
# print(a[1][2])