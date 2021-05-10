import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt

def otsuthres(histcounts, num_bins=256):
    p = histcounts / np.sum(histcounts)
    omega = np.cumsum(p)
    mu   = np.cumsum(p * np.arange(1,num_bins + 1).T)
    mu_t = mu[-1]

    sigma_b_squared = (mu_t * omega - mu)**2 / (omega * (1 - omega))
    sigma_b_squared = np.nan_to_num(sigma_b_squared, nan=-1)

    maxval = max(sigma_b_squared)
    print(maxval)
    if maxval != -1:
        idx = np.mean(np.argwhere(sigma_b_squared == maxval))
        t = (idx - 1) / (num_bins - 1)
    else:
        t = 0
    
    return t

def graythres_otsu(img):
    hist,bins = np.histogram(img.flatten(), bins=np.arange(0, 256 + 1))
    return otsuthres(hist)

if __name__ == '__main__':
    # input.JPG was taken by EOS 6D MarkII, the size is 6240 * 4160 pixels
    im_input   = im.open("./ImgOut/2-2_Grayscale.JPG")

    # convert to np
    im_input = np.array(im_input, dtype=int) # Expect uint8
    
    # Otsu's level
    threshold_otsu = (graythres_otsu(im_input)*255).astype('int64')
    im_out = (im_input > threshold_otsu).astype('float64')

    # imsave
    plt.imsave('./ImgOut/2-4_Binarization_of_Otsu.JPG', im_out, cmap = "gray")