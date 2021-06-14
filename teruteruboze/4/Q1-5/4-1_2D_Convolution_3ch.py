import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage

class Image:
    def __init__(self, path):
        self.path = path
        self.img  = self.imread()
        self.PILImage2numpy() # convert self.img to numpy

    def imread(self):
        return PILImage.open(self.path)

    def imsave(self, path):
        self.Clip()
        plt.imsave(path, self.img)

    def Clip(self):
        self.img[self.img > 1] = 1
        self.img[self.img < 0] = 0

    def PILImage2numpy(self):
        self.img = np.array(self.img, dtype=int)
        self.img = self.img / 255

class FilterKernel:
    def __init__(self, Filter, bias):
        self.W = Filter
        self.b = bias

        try:
            self.FN, self.FCh, self.FH, self.FW = self.W.shape
        except ValueError:
            raise Exception('Filter must have shape =  (FN, Ch, H, W)')

    def conv2D(self, x):
        try:
            Ch, H, W = x.shape
        except ValueError:
            raise Exception('Input data must have shape =  (Ch, H, W)')

        OH = 1 + int((H - self.FH))
        OW = 1 + int((W - self.FW))

        Filter = self.W.reshape(self.FN, -1).T # Filterのreshape
        x      = self.im2col(x, OH, OW, Ch) # xのim2col化
        Out    = (x @ Filter) + self.b # 行列積で計算可

        return Out.reshape(OH, OW, -1).transpose(2, 0, 1)

    def im2col(self, im, OH, OW, Ch):
        col = np.empty([Ch, self.FH, self.FW, OH, OW])
        for ih in range(self.FH):
            for iw in range(self.FW):
                col[:, ih, iw, :, :] = im[:, ih:ih+OH, iw:iw+OW]
        return col.transpose(3, 4, 0, 1, 2).reshape(OH*OW, -1)

if __name__ == '__main__':
    # 注意:カーネルは正方を仮定
    Kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    F = np.zeros((3, 3, Kernel.shape[0], Kernel.shape[1]))
    for i in range(Kernel.shape[0]):
        F[i,i] = Kernel
    Filter = FilterKernel(F, 0)

    A = Image('./Img/coin3ch.PNG')
    A.img = np.transpose(A.img, (2, 0, 1)) # (H, W, Ch) -> (Ch, H, W) due to conv2D
    A.img = Filter.conv2D(A.img)
    A.img = np.transpose(A.img, (1, 2, 0)) # retrive (Ch, H, W) -> (H, W, Ch) due to imsave
    A.imsave('./ImgOut/4-1_2D_Convolution.png')

    """
    x = np.array([[1, 2, 3, 0],
                  [0, 1, 2, 3],
                  [3, 0, 1, 2],
                  [2, 3, 0, 1]])
    x = np.tile(x, (3, 1, 1)) # reps
    out = Filter.conv2D(x)
    """