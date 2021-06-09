import numpy as np
from PIL import Image
from utils import *

img = np.array(Image.open('./lena.png').convert('L'))
img = img.reshape(1, 1, img.shape[0], img.shape[1])
fil_num = 1
kernel = np.array([[1,1,1], [1,1,1],[1,1,1]], np.float32)/9
kernel = kernel.reshape(fil_num, 1, kernel.shape[0], kernel.shape[1])
bias = np.zeros(fil_num)

conv = Conv2D(kernel, bias, stride=5)
conv_imgs = conv.forward(img)
output = Image.fromarray(np.uint8(conv_imgs[0,:,:,0]))
output.save('./q4_output.png')
