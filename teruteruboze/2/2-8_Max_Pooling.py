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

def max_pooling(img, h, w, c, ks=8):

    for y in range(int(h / ks)):
        for x in range(int(w / ks)):
            for ch in range(c):
                img[ks*y:ks*(y+1), ks*x:ks*(x+1), ch] = \
                np.max(img[ks*y:ks*(y+1), ks*x:ks*(x+1), ch])
    
    return img

if __name__ == '__main__':
    Input = ImageTools("./Img/imori.JPG")
    Input.to_numpy_float(), Input.getShape()

    Input.img = max_pooling(Input.img, *Input.shape)

    # imsave
    plt.imsave('./ImgOut/2-8_Max_Pooling.JPG', Input.img)