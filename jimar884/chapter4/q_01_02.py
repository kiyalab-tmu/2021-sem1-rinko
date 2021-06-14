import os
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.type_check import imag
import torch


def rgb2gray(image):
    H, W, _ = image.shape
    r = image[:,:,0].copy()
    g = image[:,:,1].copy()
    b = image[:,:,2].copy()
    ans = r*0.2126 + g*0.7152 + b*0.0722

    return ans


def conv2D(input, kernel, p_size=0, s_size=1):
    input = rgb2gray(input)
    output = padding(input, p_size=p_size)
    output = stride(input, kernel, s_size=s_size)
    return output


def padding(input, p_size=0):
    H, W = input.shape
    ans = np.zeros((H+2*p_size, W+2*p_size))
    ans[p_size:H+p_size, p_size:W+p_size] = input
    return ans


def stride(input, kernel, s_size=1):
    H, W = input.shape
    H_k, W_k = kernel.shape
    ans = np.zeros(((H-H_k)//s_size,(W-W_k)//s_size))
    tmp = input.copy()
    for h in range(0, H-H_k, s_size):
        for w in range(0, W-W_k, s_size):
            ans[h,w] = np.sum(tmp[h:h+H_k,w:w+W_k]*kernel)
    return ans


def main():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    IMG_FILE_NAME = "imori.png"
    FULL_PATH = os.path.join(BASE_PATH, IMG_FILE_NAME)
    image = img.imread(FULL_PATH, 0)
    kernel = np.array([
        [-1,-1,-1],
        [-1, 8,-1],
        [-1,-1,-1]
    ])
    a = conv2D(image, kernel)
    plt.imshow(a, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()

