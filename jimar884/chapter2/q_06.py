"""
Discretion of Color
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def decrease_color(img):
    """
    decrease color
    each channel has 4 kind of values
    """
    result_img = img.copy()

    result_img[(0 <= result_img) & (result_img < 64)] = 32
    result_img[(64 <= result_img) & (result_img < 127)] = 96
    result_img[(127 <= result_img) & (result_img < 191)] = 160
    result_img[(191 <= result_img) & (result_img < 256)] = 224
    return result_img

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
IMG_FILE_NAME = "kinkaku2.JPG"
FULL_PATH = os.path.join(BASE_PATH, IMG_FILE_NAME)
img = cv2.imread(FULL_PATH)

img2 = decrease_color(img)
# show image
cv2.imshow('original & decrease color', cv2.hconcat([img, img2]))
cv2.waitKey(0)