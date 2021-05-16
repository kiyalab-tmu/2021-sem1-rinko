import cv2 as cv
import numpy as np

img=cv.imread("imori.jpg")
out=np.zeros((15,15,3))
h,w,c= img.shape
i1=0
j1=0
for i in range(0,h-8,8):
    for j in range(0,w-8,8):
        sum = [0, 0, 0]
        max=[0,0,0]
        for x in range(8):
            for y in range(8):
                   if (max[0]<img[i+x,j+y][0]):
                       max[0]=img[i+x,j+y][0]
                   if (max[1] < img[i + x, j + y][1]):
                       max[1]=img[i + x, j + y][1]
                   if (max[2] < img[i + x, j + y][2]):
                       max[2]=img[i + x, j + y][2]
        out[i1,j1]=max
        j1+=1
    i1+=1
    j1=0
cv.imshow("result",out)
cv.imwrite("out2.jpg", out)
cv.waitKey(0)


