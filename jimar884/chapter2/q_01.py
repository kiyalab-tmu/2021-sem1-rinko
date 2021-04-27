"""
RGBをBGRに入れ替える
"""

import os
import cv2

def rgb2bgr(img):
    """
    cv2.imreadはbgrのチャネル構成
    rgbに入れ替えて返す
    """
    blue = img[:, :, 0].copy()
    green = img[:, :, 1].copy()
    red = img[:, :, 2].copy()
    img[:, :, 0] = red
    img[:, :, 1] = green
    img[:, :, 2] = blue

    return img

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG_FILE_NAME = "kinkaku2.JPG"
FULL_PATH = os.path.join(BASE_PATH, IMG_FILE_NAME)
img = cv2.imread(FULL_PATH)

img2 = rgb2bgr(img.copy())

img3 = cv2.hconcat([img, img2])
cv2.imshow('compare 2 images', img3)
cv2.waitKey(0)
