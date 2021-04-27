"""
Max Pooling
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def max_pooling(img):
    """
    input BGR image
    perform an Max Pooling
    return image
    """
    result_img = img.copy()
    heignt, width, _ = result_img.shape
    for h in range(0, heignt, 8):
        for w in range(0, width, 8):
            result_img[h:h+8, w:w+8, 0] = np.max(result_img[h:h+8, w:w+8, 0])
            result_img[h:h+8, w:w+8, 1] = np.max(result_img[h:h+8, w:w+8, 1])
            result_img[h:h+8, w:w+8, 2] = np.max(result_img[h:h+8, w:w+8, 2])
    result_img[(heignt//8)*8:heignt, :, :] = 0
    result_img[:, (width//8)*8:width, :] = 0
    return result_img

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG_FILE_NAME = "kinkaku2.JPG"
FULL_PATH = os.path.join(BASE_PATH, IMG_FILE_NAME)
img = cv2.imread(FULL_PATH)

img2 = max_pooling(img)
# show image
cv2.imshow('original & pooling', cv2.hconcat([img, img2]))
cv2.waitKey(0)