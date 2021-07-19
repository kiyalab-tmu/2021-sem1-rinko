import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt

def channel_swapping(img):
    temp = img[:,:,0].copy()
    img[:,:,0] = img[:,:,2].copy()
    img[:,:,2] = temp.copy()
    return img


if __name__ == '__main__':
    # input.JPG was taken by EOS 6D MarkII, the size is 6240 * 4160 pixels
    im_input   = im.open("./Img/imori.JPG")

    # convert to np
    im_input = np.array(im_input, dtype=int) # Expect uint8
    im_input = im_input / 255 #uint8 to float
    im_out   = channel_swapping(im_input)

    # imsave
    plt.imsave('./ImgOut/2-1_Channel_Swapping.JPG', im_out)

