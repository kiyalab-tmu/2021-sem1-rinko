"""
RGBをBGRに入れ替える
"""

import os
import cv2
import numpy as np
from math import floor

# def apply_threshold(value, bitdepth):
#     return floor(floor(value / 2**(8 - bitdepth)) / (2**bitdepth - 1) * 255)

def apply_threshold(value, bitdepth):
    return floor(value/2**(8-bitdepth))*floor(255/(2**bitdepth-1))

def minmax(v):
  if v > 255:
    v = 255
  if v < 0:
    v = 0
  return v


def Floyd_Steinberg(imgs, out_color):
    result_img = imgs.copy()
    H, W, C = result_img.shape
    th = 2**8 // out_color
    for c in range(C):
        for h in range(H):
            for w in range(W):
                old = result_img[h, w, c]
                # new = (old // th) * th + th // 2
                new = apply_threshold(old, 1)
                result_img[h, w, c] = new
                error = old - new

                if w < W-1:
                    result_img[h, w+1, c] = minmax(result_img[h, w+1, c] + error*7/16)
                if h < H-1 and w > 0:
                    result_img[h+1, w-1, c] = minmax(result_img[h+1, w-1, c] + error*3/16)
                if h < H-1:
                    result_img[h+1, w, c] = minmax(result_img[h+1, w, c] + error*5/16)
                if h < H-1 and w < W-1:
                    result_img[h+1, w+1, c] = minmax(result_img[h+1, w+1, c] + error*1/16)
    result_img.astype(np.uint8)

    return result_img

def grayscale(img):
    """
    グレースケール化した画像を返す
    """
    img[:, :, 0] = img[:, :, 0].copy() * 0.0722
    img[:, :, 1] = img[:, :, 1].copy() * 0.7152
    img[:, :, 2] = img[:, :, 2].copy() * 0.2126
    img = img.astype(np.uint8)

    return img

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
print(img.shape)
# img2 = grayscale(img.copy())
img2 = Floyd_Steinberg(img.copy(), 4)

img3 = cv2.hconcat([img, img2])
cv2.imshow('compare 2 images', img3)
cv2.waitKey(0)
