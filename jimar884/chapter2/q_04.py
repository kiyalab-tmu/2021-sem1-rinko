"""
大津の2値化
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale(img):
    """
    input BGR img
    return grayscale image
    """
    blue = img[:, :, 0].copy()
    green = img[:, :, 1].copy()
    red = img[:, :, 2].copy()
    result_img = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    result_img = result_img.astype(np.uint8)

    return result_img

def binary_image(img, thread=128):
    """
    input BGR img
    return binary image
    """
    result_img = grayscale(img)
    result_img[result_img < thread] = 0
    result_img[result_img >= thread] = 255

    return result_img

def ootsu(img):
    """
    input BGR img
    return binary image (thread is decided by otsu)
    """
    gray_img = grayscale(img)
    max_sigma = 0
    best_th = 0
    for th in range(1, 255):
        # 各クラスに属する画素
        w0, w1 = gray_img[gray_img < th], gray_img[gray_img >= th]
        # 各クラスの画素値の平均
        m0, m1 = np.mean(w0), np.mean(w1)
        # クラス間分散
        sigma = (m0 - m1)**2 * len(w0) * len(w1) / (len(w0) + len(w1))**2
        if sigma > max_sigma:
            max_sigma = sigma
            best_th = th
    result_img = binary_image(img, thread=best_th)
    print(best_th)
    return result_img


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG_FILE_NAME = "kinkaku2.JPG"
FULL_PATH = os.path.join(BASE_PATH, IMG_FILE_NAME)
# color image
img = cv2.imread(FULL_PATH)

# gray image
img2 = grayscale(img)
# ootsu image
img3 = ootsu(img)
# ravel:1次元化
# bins:分割数
# rwidth:棒の幅
# range:横軸の目盛り
plt.hist(img2.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.xlabel('value')
plt.ylabel('appearance')
plt.show()
#show image
cv2.imshow('original & otsu', cv2.hconcat([img, cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)]))
cv2.waitKey(0)
