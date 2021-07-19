import cv2 as cv
import numpy as np
img=cv.imread("imori.jpg").astype(np.float)
r=img[:,:,0].copy()
g=img[:,:,1].copy()
b=img[:,:,2].copy()
grey=0.2126*r + 0.7152*g + 0.0722*b
cv.imshow("result",grey.astype(np.uint8))
cv.waitKey(0)