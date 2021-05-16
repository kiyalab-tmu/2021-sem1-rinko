import cv2 as cv
import numpy as np

img=cv.imread("imori.jpg")
img[np.where((img>=0) & (img<63))]=32
img[np.where((img>=63) & (img<127))]=96
img[np.where((img>=127) & (img<191))]=160
img[np.where((img>=191) & (img<256))]=224

cv.imshow("result",img)
cv.waitKey(0)