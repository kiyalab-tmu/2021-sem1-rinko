import numpy as np

class FilterKernel:
    def __init__(self, Filter, bias):
        self.W = Filter
        self.b = bias

    def conv2D(self, x):
        self.FH, self.FW = self.W.shape
        H,   W = x.shape

        OH = 1 + int((H - self.FH))
        OW = 1 + int((W - self.FW))
        
        Knl = self.W.reshape([1, self.FH*self.FW]) # Filterのreshape
        x   = self.im2col(x, OH, OW) # xのim2col化
        Out = (Knl @ x) + self.b # 計算

        return Out.reshape([OH, OW])

    def AvePooling2D(self, x, FH, FW):
        self.FH = FH
        self.FW = FW
        print(self.FW, self.FH)
        H,   W = x.shape

        OH = 1 + int((H - FH))
        OW = 1 + int((W - FW))
        
        x   = self.im2col(x, OH, OW) # xのim2col化
        print(x.shape)
        Out = np.mean(x, axis=0)

        return Out.reshape([OH, OW])

    def MaxPooling2D(self, x, FH, FW):
        self.FH = FH
        self.FW = FW
        print(self.FW, self.FH)
        H,   W = x.shape

        OH = 1 + int((H - FH))
        OW = 1 + int((W - FW))
        
        x   = self.im2col(x, OH, OW) # xのim2col化
        print(x.shape)
        Out = np.max(x, axis=0)

        return Out.reshape([OH, OW])

    # im2colのためのreshpeは，自作が必要（通常のreshapeでは置換できない）：こっちは自分で考えた方，重い．
    # こちらのコードは非常に計算コストが高い
    # ref: https://qiita.com/kuroitu/items/35d7b5a4bde470f69570#im2colの動作と初期の実装
    def _im2col(self, im, OH, OW):
        col = np.empty([self.FH*self.FW, OH*OW])
        for ih in range(OH):
            for iw in range(OW):
                col[:, iw + ih*OW] = im[ih:ih+self.FH, iw:iw+self.FW].reshape(-1)
        return col

    # こうすると軽量化される，先達あらまほしきことなり．：_im2colに対して，**軸が反転してると思えば良い**
    def im2col(self, im, OH, OW):
        col = np.empty([self.FH, self.FW, OH, OW])
        for ih in range(self.FH):
            for iw in range(self.FW):
                col[ih, iw, :, :] = im[ih:ih+OH, iw:iw+OW]
        return col.reshape(self.FH*self.FW, OH*OW)

if __name__ == '__main__':
    F = np.array([[2, 0, 1],
                  [0, 1, 2],
                  [1, 0, 2]])
    Filter = FilterKernel(F, 0)

    x = np.array([[1, 2, 3, 0],
                  [0, 1, 2, 3],
                  [3, 0, 1, 2],
                  [2, 3, 0, 1]])
    Conv    = Filter.conv2D(x)
    AvePool = Filter.AvePooling2D(x, 3, 3)
    MaxPool = Filter.MaxPooling2D(x, 3, 3)

    print(Conv)
    print(AvePool)
    print(MaxPool)