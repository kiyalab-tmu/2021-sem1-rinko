import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt

class ImageTools:
    def __init__(self, path):
        self.img = im.open(path)
        self.shape = self.getShape()

    # convert to numpy
    def to_numpy_float(self):
        self.img = np.array(self.img, dtype=int) # Expect uint8
        self.img = self.img / 255 #uint8 to float

    def to_numpy_int(self):
        self.img = np.array(self.img, dtype=int) # Expect uint8

    def getShape(self):
        temp_img   = np.array(self.img, dtype=int) # convert to mesure
        return temp_img.shape

def im2ZeroPadding(img, h, w, c, ks):
    pad = ks // 2 # カーネルサイズによってパディングに必要なマス数が変わる
    im_zero = np.zeros((h + pad * 2, w + pad * 2, c), dtype=np.float) # 上下左右に1マスずつ（計2個）パディング
    im_zero[pad: pad + h, pad: pad + w] = img.copy().astype(np.float)
    return im_zero

def Gaussian_Filter(ks, sigma):
    pad = ks // 2 # カーネルサイズによってパディングに必要なマス数が変わる
    Kernel = np.zeros((ks, ks), dtype=np.float)
    for x in range(-pad, ks-pad):
        for y in range(-pad, ks-pad):
            Kernel[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    Kernel /= (2 * np.pi * sigma ** 2)
    Kernel /= Kernel.sum()
    return Kernel


def applyGaussian_Filter(img, h, w, c, ks=3, sigma=1.3):

    img = im2ZeroPadding(img, h, w, c, ks)
    Kernel = Gaussian_Filter(ks, sigma)

    for y in range(h):
        for x in range(w):
            for ch in range(c):
                img[ks // 2 + y, ks // 2 + x, ch] = \
                np.sum(Kernel*img[y:y + ks, x:x + ks, ch])
    
    return img

if __name__ == '__main__':
    Input = ImageTools("./Img/imori_noise.JPG")
    Input.to_numpy_float(), Input.getShape()

    Input.img = applyGaussian_Filter(Input.img, *Input.shape)

    # imsave
    plt.imsave('./ImgOut/2-9_Gaussian_Filter.JPG', Input.img)