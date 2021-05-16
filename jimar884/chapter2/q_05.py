"""
RGB -> HSV & HSV -> RGB
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def bgr2hsv(img):
    """
    input BGR img
    BGR => HSV
    return HSV image
    """
    result_img = np.zeros_like(img, dtype=np.float32)
    # pixel values : [0, 255] -> [0, 1]
    img_sub = img.copy() / 255
    # if the value of blue channel is minimum, get 0.
    # blue -> 0, green -> 1, blue -> 2
    min_channel = np.argmin(img_sub, axis=2)
    # get the minimum value of each pixel (thre value of blue, green or red)
    min_v = np.min(img_sub, axis=2).copy()
    # get the maximum values of each pixel
    max_v = np.max(img_sub, axis=2).copy()

    # Hue : [0, 360] (sometimes, it is normalized to [0, 100])
    index = np.where(min_v == max_v)
    result_img[..., 0][index] = 0
    index = np.where(min_channel == 0)     # min is blue
    result_img[..., 0][index] = 60 * (img_sub[..., 1][index] - img_sub[..., 2][index]) / (max_v[index] - min_v[index]) + 60
    index = np.where(min_channel == 1)     # min is green
    result_img[..., 0][index] = 60 * (img_sub[..., 2][index] - img_sub[..., 0][index]) / (max_v[index] - min_v[index]) + 300
    index = np.where(min_channel == 2)     # min is red
    result_img[..., 0][index] = 60 * (img_sub[..., 0][index] - img_sub[..., 1][index]) / (max_v[index] - min_v[index]) + 180
    # Saturation Chroma : [0, 1]
    result_img[..., 1] = max_v - min_v
    # Value Brightness : [0, 1]
    result_img[..., 2] = max_v

    return result_img


def hsv2bgr(img):
    """
    input HSV img
    HSV -> BGR
    return BGR image
    """
    result_imgae = np.zeros_like(img)
    H = img[..., 0].copy()
    S = img[..., 1].copy()
    V = img[..., 2].copy()

    C = S
    H_ = H / 60
    X = C * (1 - np.abs( H_ % 2 - 1))
    Z = np.zeros_like(H)

    vals = [[C,X,Z], [X,C,Z], [Z,C,X], [Z,X,C], [X,Z,C], [C,Z,X]]

    for i in range(6):
        ind = np.where((i <= H_) & (H_ < (i+1)))
        result_imgae[..., 2][ind] = (V - C)[ind] + vals[i][0][ind]
        result_imgae[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
        result_imgae[..., 0][ind] = (V - C)[ind] + vals[i][2][ind]

    result_imgae = (result_imgae * 255).astype(np.uint8)

    return result_imgae


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG_FILE_NAME = "kinkaku2.JPG"
FULL_PATH = os.path.join(BASE_PATH, IMG_FILE_NAME)
img = cv2.imread(FULL_PATH)

# hsv image
img2 = bgr2hsv(img)
# invert the Hue
img2[:, :, 0] = (img2[:, :, 0] + 180) % 360
# rgb imgae (not original)
img3 = hsv2bgr(img2)
# show image
cv2.imshow('original & inversed H', cv2.hconcat([img, img3]))
cv2.waitKey(0)