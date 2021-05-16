"""
グレースケール化
"""

import os
import cv2
import numpy as np

def grayscale(img):
    """
    グレースケール化した画像を返す
    """
    blue = img[:, :, 0].copy()
    green = img[:, :, 1].copy()
    red = img[:, :, 2].copy()
    result = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    result = result.astype(np.uint8)

    return result

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG_FILE_NAME = "kinkaku2.JPG"
FULL_PATH = os.path.join(BASE_PATH, IMG_FILE_NAME)
img = cv2.imread(FULL_PATH)

img2 = grayscale(img.copy().astype(np.float))

# cvtColorを使うことでhconcatできる
img3 = cv2.hconcat([img, cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)])
cv2.imshow('compare 2 images', img3)
cv2.waitKey(0)
