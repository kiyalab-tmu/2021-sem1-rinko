import cv2 as cv

img=cv.imread("imori.jpg")
r=img[:,:,0].copy()
b=img[:,:,2].copy()
img[:,:,0]=b
img[:,:,2]=r
cv.imshow("result",img)
cv.waitKey(0)