import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # input.JPG was taken by EOS 6D MarkII, the size is 6240 * 4160 pixels
    im_input   = im.open("./ImgOut/2-2_Grayscale.JPG")
    threshold  = 128

    # convert to np
    im_input = np.array(im_input, dtype=int) # Expect uint8
    im_out = (im_input > threshold).astype('float64')

    # imsave
    plt.imsave('./ImgOut/2-3_Binarization.JPG', im_out, cmap = "gray")