import cv2 as cv
import numpy as np
img=cv.imread("imori.jpg")
r=img[:,:,0].copy()
g=img[:,:,1].copy()
b=img[:,:,2].copy()
grey=0.2126*r + 0.7152*g + 0.0722*b
grey[grey<128]=0
grey[grey>=128]=255

cv.imshow("result",grey)
cv.waitKey(0)

