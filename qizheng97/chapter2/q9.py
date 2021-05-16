import cv2 as cv
import numpy as np

img=cv.imread("imori_noise.jpg")
img=cv.GaussianBlur(img,(3,3),1.3)
cv.imshow("result",img)
cv.waitKey(0)