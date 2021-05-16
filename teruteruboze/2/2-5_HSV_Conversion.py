import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import cv2

class ImageTools:
    def __init__(self, path):
        self.img = cv2.imread(path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.imsize = [self.img.shape[0], self.img.shape[1], self.img.shape[2]]

    # convert to numpy
    def to_numpy_float(self):
        self.img = np.array(self.img, dtype=int) # Expect uint8
        self.img = self.img / 255 #uint8 to float

    def rgb2hsv(self):
        self.img = cv2.cvtColor(self.img,cv2.COLOR_RGB2HSV)

    def hsv2rgb(self):
        self.img = cv2.cvtColor(self.img,cv2.COLOR_HSV2RGB)

    # 自作中
    def _rgb2hsv(self):
        h, w, c  = self.img.shape[0], self.img.shape[1], self.img.shape[2]
        self.img = self.img.reshape(h * w, c)
        V = self.img.max(axis=1)
        print(V.shape)

def invHue(img):
    H = img[:,:,0]
    H += 90
    H[H>=180] -= 180
    img[:,:,0] = H
    return img

if __name__ == '__main__':
    # input.JPG was taken by EOS 6D MarkII, the size is 6240 * 4160 pixels
    Input = ImageTools("./Img/input.JPG")
    Input.rgb2hsv()
    Input.img = invHue(Input.img)
    Input.hsv2rgb()

    # imsave
    plt.imsave('./ImgOut/2-5_HSV_Conversion.JPG', Input.img)