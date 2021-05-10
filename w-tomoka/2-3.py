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

#th, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
img_bin = binary(img_gray, 127)

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