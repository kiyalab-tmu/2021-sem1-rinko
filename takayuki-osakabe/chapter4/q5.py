import numpy as np
from PIL import Image
from layers import *

img = np.array(Image.open('./lena.png').convert('L'))
img = img.reshape(1, 1, img.shape[0], img.shape[1])
PH = 5
PW = 5
stride = 1

max_pool = Pooling(PH, PW, stride=stride, mode='max')
avg_pool = Pooling(PH, PW, stride=stride, mode='avg')
max_img = max_pool.forward(img)
avg_img = avg_pool.forward(img)
max_img = Image.fromarray(np.uint8(max_img[0,:,:,0]))
avg_img = Image.fromarray(np.uint8(avg_img[0,:,:,0]))

max_img.save('./q5_output_max.png')
avg_img.save('./q5_output_avg.png')
