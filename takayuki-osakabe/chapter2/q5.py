import cv2
import numpy as np

def rgb2hsv(pixel):
    b,g,r = pixel
    r,g,b = r/255.0, g/255.0, b/255.0
    cmax = max(r,g,b)
    cmin = min(r,g,b)
    diff = cmax - cmin

    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = (60 * ((g-b)/diff) + 360) % 360
    elif cmax == g:
        h = (60 * ((b-r)/diff) + 120) % 360
    elif cmax == b:
        h = (60 * ((r-g)/diff) + 240) % 360
    if cmax == 0:
        s = 0
    else:
        s = (diff/cmax)*100
    v = cmax*100
    return h, s, v

img = cv2.imread('./sample.jpg')

'''
output = np.empty((img.shape))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        h,s,v = rgb2hsv(img[i,j,:])
        output[i,j,:] = [h,s,v]
'''
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv[:,:,0] += 180

output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imwrite('./q5_output.png', output)
