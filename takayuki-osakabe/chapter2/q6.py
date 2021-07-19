import cv2
import numpy as np

def discretization(img):
    img[(img >= 0) == (img < 63)] = 32
    img[(img >= 63) == (img < 127)] = 96
    img[(img >= 127) == (img < 191)] = 160
    img[(img >= 191) == (img < 224)] = 224

    return img


img = cv2.imread("./sample.jpg")

output = discretization(img)

cv2.imwrite('./q6_output.png', output)
