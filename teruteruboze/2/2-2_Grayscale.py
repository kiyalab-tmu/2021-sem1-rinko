import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # input.JPG was taken by EOS 6D MarkII, the size is 6240 * 4160 pixels
    im_input   = im.open("./Img/imori.JPG")

    # convert to np
    im_input = np.array(im_input, dtype=int) # Expect uint8
    im_input = im_input / 255 #uint8 to float
    h, w, c  = im_input.shape[0], im_input.shape[1], im_input.shape[2]
    im_input = im_input.reshape(h * w, c) # reshape for matrix dot
    
    # prep color XFR matrix
    M = np.array([0.2126, 0.7152, 0.0722]).reshape(3,1)

    # perfrom color XFR
    im_out = im_input @ M
    im_out = im_out.reshape(h, w)

    # imsave
    plt.imsave('./ImgOut/2-2_Grayscale.JPG', im_out, cmap = "gray")