import numpy as np
from PIL import Image
from utils import *

img = np.array(Image.open('./lena.png'))
img = img.reshape(1, img.shape[2], img.shape[0], img.shape[1])
fil_num = 3
k_size = 3
kernel = np.random.randn(fil_num, img.shape[1], k_size, k_size)
bias = np.zeros(fil_num)

conv = Conv2D(kernel, bias)
conv_imgs = conv.forward(img)

output = Image.fromarray(np.uint8(conv_imgs[0,:,:,:]))
output.save('./q1_output.png')
