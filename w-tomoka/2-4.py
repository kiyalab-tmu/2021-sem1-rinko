import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

img_orig = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_512x512.png')

img_gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)

def binary(img, th):
    bin_image = img.copy()
    bin_image[img < th] = 0
    bin_image[img >= th] = 255
    return bin_image

def otsu_thresh(img):
    max_vari = -1
    max_th = 0
    for th in range(1, 254):
        m0 = img[img <= th].mean() #class 0 の平均
        m1 = img[img > th].mean() #class 1 の平均
        w0 = img[img <= th].size #class 0 の画素数
        w1 = img[img > th].size #class 1 の画素数
        vari = w0 * w1 * (m0 - m1)**2
        if vari > max_vari:
            max_vari = vari
            max_th = th
        
    img = binary(img, max_th)
    return max_th, img

#th, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
th, img_bin = otsu_thresh(img_gray)

print('threshould :', th)

plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.title('input')
plt.imshow(img_orig)
plt.subplot(1, 3, 2)
plt.title('gray')
plt.imshow(img_gray, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('answer')
plt.imshow(img_bin, cmap='gray')
plt.show()