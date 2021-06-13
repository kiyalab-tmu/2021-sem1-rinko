import numpy as np
from numpy.core.fromnumeric import transpose
import numpy.matlib as npmat
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

# 3 Channel 対応済
class FilterKernel:
    def __init__(self, Filter, bias):
        self.W = Filter
        self.b = bias

        if len(self.W.shape) == 2:
            self.W = np.tile(self.W, (3, 1, 1)) # If Ch = 1, then do repmat to 3

        self.FCh, self.FH, self.FW = self.W.shape
        self.W = self.W.reshape([self.FCh, 1, self.FH*self.FW]) # Filterのreshape

    def conv2D(self, x):
        H, W, Ch = x.shape

        OH = 1 + int((H - self.FH))
        OW = 1 + int((W - self.FW))
        
        x   = self._im2col(x, OH, OW) # xのim2col化
        Out = np.empty([self.FCh, OH*OW])
        # Per channel
        for iCh in range(self.FCh):
            Out[iCh,:] = (self.W[iCh,:,:] @ x[iCh,:,:]) + self.b # 計算
        Out = np.transpose(Out, (1, 0))
        return Out.reshape([OH, OW, self.FCh])

    # im2colのためのreshpeは，自作が必要（通常のreshapeでは置換できない）
    # こちらのコードは非常に計算コストが高い
    # ref: https://qiita.com/kuroitu/items/35d7b5a4bde470f69570#im2colの動作と初期の実装
    def _im2col(self, x, OH, OW):
        out = np.empty([self.FCh, self.FH*self.FW, OH*OW])
        for iCh in range(self.FCh):
            for ih in range(OH):
                for iw in range(OW):
                    out[iCh, :,iw + ih*OW] = x[ih:ih+self.FH, iw:iw+self.FW, iCh].reshape(-1)

        return out

if __name__ == '__main__':
    A = Image('./Img/coin3ch.PNG')
    print(A.img.shape)

    F = np.array([[-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1]])
    Filter = FilterKernel(F, 0)

    """
    x = np.array([[1, 2, 3, 0],
                  [0, 1, 2, 3],
                  [3, 0, 1, 2],
                  [2, 3, 0, 1]])
    x = np.tile(x, (3, 1, 1))
    x = np.transpose(x, (1, 2, 0))
    """

    A.img = Filter.conv2D(A.img)
    A.imsave('./ImgOut/4-2_Edge_detection.png')