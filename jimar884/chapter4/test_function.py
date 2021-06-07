import os
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.type_check import imag
import torch

class Layer2D():
    def __init__(self, kernel=None, bias=None):
        if not(kernel==None):
            self.kernel = kernel
        else:
            self.kernel = torch.rand((3, 3))
        if not(bias==None):
            self.bias = bias
        else:
            self.bias = torch.zeros((3, 3))


def rgb2gray(image):
    H, W, _ = image.shape
    r = image[:,:,0].copy()
    g = image[:,:,1].copy()
    b = image[:,:,2].copy()
    ans = r*0.2126 + g*0.7152 + b*0.0722

    return ans


def filter(input, layer):
    ans = 0
    for i in range(len(layer.kernel)):
        for j in range(len(layer.kernel)):
            ans += layer.kernel[i, j]*input[i, j] + layer.bias[i,j]
    return ans


def conv2D(input, layer):
    H, W = input.shape
    k_size = len(layer.kernel)
    s = k_size//2
    output = torch.zeros((H-k_size+1, W-k_size+1))
    for h in range(s, H-s-1):
        for w in range(s, W-s-1):
            output[h-s, w-s] = (
                filter(input[h-s:h+s+1, w-s:w+s+1], layer)
            )
    output = (output-output.min()) / (output.max()-output.min())
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
    gray_image = rgb2gray(image)

    # print(gray_image.shape)
    # plt.imshow(gray_image, cmap="gray")
    # plt.show()
    kernel = np.array([
        [-1,-1,-1],
        [-1, 8,-1],
        [-1,-1,-1]
    ])
    # layer = Layer2D(kernel=kernel)
    # a = conv2D(torch.tensor(gray_image), layer)
    # print(type(a))
    # a = np.array(a)
    a = padding(gray_image, p_size=2)
    a = stride(a, kernel=kernel)
    plt.imshow(a, cmap="gray")
    plt.show()



if __name__ == "__main__":
    main()

