"""
Gaussian Filter
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_filter(img, sigma=1.3, k_size=3):
    """
    input BGR image, standard deviation and kernel size(k_seze x k_size)
    return BGR image filtered by Gaussian_filter
    """
    heignt, width, channel = img.shape

    pad = k_size // 2
    result_img = np.zeros((heignt+pad*2, width+pad*2, channel), dtype=np.float)
    result_img[pad:pad+heignt, pad:pad+width] = img.copy().astype(np.float)

    kernel = np.zeros((k_size, k_size), dtype=np.float)
    for x in range(-pad, -pad+k_size):
        for y in range(-pad, -pad+k_size):
            kernel[y+pad, x+pad] = np.exp(-(x**2 + y**2) / (sigma**2))
    kernel /= (2 * np.pi * sigma * sigma)
    kernel /= kernel.sum()

    tmp = result_img.copy()

    for y in range(heignt):
        for x in range(width):
            for c in range(channel):
                result_img[y+pad, x+pad, c] = np.sum(kernel * tmp[y:y+k_size, x:x+k_size, c])
    result_img = result_img.astype(np.uint8)
    return result_img


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG_FILE_NAME = "kinkaku2.JPG"
FULL_PATH = os.path.join(BASE_PATH, IMG_FILE_NAME)
img = cv2.imread(FULL_PATH)

img2 = gaussian_filter(img)

cv2.imshow("gaussian", img2)
cv2.waitKey(0)




