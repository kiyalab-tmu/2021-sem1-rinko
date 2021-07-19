import cv2 as cv
import numpy as np

img=cv.imread("imori.jpg")
r=img[:,:,0].copy()
g=img[:,:,1].copy()
b=img[:,:,2].copy()
grey=0.2126*r + 0.7152*g + 0.0722*b
[a,b,c]=img.shape
max=0
maxi=0
for i in range(255):
    p0=grey[grey<=i]
    p1=grey[grey>i]
    if (len(p0)==0) or (len(p1)==0):
        continue
    m0=np.mean(p0)
    m1=np.mean(p1)
    s0=np.var(p0)
    s1=np.var(p1)
    w0=len(p0)/(a*b)
    w1=1-w0
    sw=w0*s0+w1*s1
    sb=w0*w1*((m0-m1)**2)
    st=sw+sb
    if (st>max):
        max=st
        maxi=i
print(maxi)
grey[grey<=maxi]=0
grey[grey>maxi]=255
cv.imshow("result",grey)
cv.waitKey(0)



