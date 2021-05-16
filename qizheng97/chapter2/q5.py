import cv2 as cv
import numpy as np

img=cv.imread("imori.jpg")
img=img/255

hsv=np.zeros_like(img,dtype=np.float32)
max=np.max(img,axis=2).copy()
min=np.min(img,axis=2).copy()
argmin=np.argmin(img,axis=2)

hsv[:,:,0][np.where(max==min)]=0
index=np.where(argmin==0)
hsv[:,:,0][index]=60*((img[:,:,2][index]-img[:,:,1][index])/(max[index]-min[index]))+180
index=np.where(argmin==1)
hsv[:,:,0][index]=60*((img[:,:,0][index]-img[:,:,2][index])/(max[index]-min[index]))+300
index=np.where(argmin==2)
hsv[:,:,0][index]=60*((img[:,:,1][index]-img[:,:,0][index])/(max[index]-min[index]))+60

hsv[:,:,1]=max.copy()-min.copy()

hsv[:,:,2]=max.copy()

hsv[:,:,0]=(hsv[:,:,0]+180)%360

h=hsv[:,:,0]
s=hsv[:,:,1]
v=hsv[:,:,2]

c=s
h1=h/60
x=c*(1-np.abs(h1%2-1))

rgb=np.zeros_like(img)

z=np.zeros_like(c)
temp=[[c,x,z],[x,c,z],[z,c,x],[z,x,c],[x,z,c],[c,z,x]]

for i in range(6):
    index = np.where((i <= h1) & (h1 < i+1))
    rgb[:,:,0][index]=(v-c)[index]+temp[i][0][index]
    rgb[:,:,1][index] = (v - c)[index] + temp[i][1][index]
    rgb[:,:,2][index] = (v - c)[index] + temp[i][2][index]

rgb=np.clip(rgb,0,1)
rgb=(rgb*255).astype(np.uint8)
cv.imshow("result",rgb)
cv.waitKey(0)



