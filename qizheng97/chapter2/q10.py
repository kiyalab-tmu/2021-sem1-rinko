import cv2 as cv
import numpy as np

img=cv.imread("imori_noise.jpg")
img=cv.medianBlur(img,3)
cv.imshow("result",img)
cv.waitKey(0)