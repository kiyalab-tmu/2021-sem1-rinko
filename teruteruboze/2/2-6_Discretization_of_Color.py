import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt

class ImageTools:
    def __init__(self, path):
        self.img = im.open(path)

    # convert to numpy
    def to_numpy_float(self):
        self.img = np.array(self.img, dtype=int) # Expect uint8
        self.img = self.img / 255 #uint8 to float

    def to_numpy_int(self):
        self.img = np.array(self.img, dtype=int) # Expect uint8

    def getShape(self):
        self.shape = [self.img.shape[0], self.img.shape[1], self.img.shape[2]]

def Discretization(img):
    img[(0<=img)&(img<63)]    = 32
    img[(63<=img)&(img<127)]  = 96
    img[(127<=img)&(img<191)] = 160
    img[(191<=img)&(img<256)] = 224
    return img

# これすごい
def Optimized_Discretization(img):
    img = img // 64 * 64 + 32
    return img

if __name__ == '__main__':
    # input.JPG was taken by EOS 6D MarkII, the size is 6240 * 4160 pixels
    Input = ImageTools("./Img/input.JPG")
    Input.to_numpy_int()
    Input.img = Discretization(Input.img)
    Input.to_numpy_float()

    # imsave
    plt.imsave('./ImgOut/2-6_Discretization_of_Color.JPG', Input.img)