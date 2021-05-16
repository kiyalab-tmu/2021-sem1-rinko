import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

img_orig = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_512x512.png')

def quantize(img):
    _image = img.copy()
    _image[img < 63] = 32
    _image[img >= 63] = 96
    _image[img >= 127] = 160
    _image[img >= 191] = 224
    return _image

#th, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
img_q = quantize(img_orig)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title('input')
plt.imshow(img_orig)
plt.subplot(1, 2, 2)
plt.title('answer')
plt.imshow(img_q)
plt.show()