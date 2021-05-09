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
        for x in range(8):
            for y in range(8):
                   sum+=img[i+x,j+y]
        out[i1,j1]=sum/64
        j1+=1
    i1+=1
    print(i1,j1)
    j1=0
cv.imshow("result",out)
cv.imwrite("out.jpg", out)
cv.waitKey(0)


