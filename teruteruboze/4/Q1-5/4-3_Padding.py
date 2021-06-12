import numpy as np

class FilterKernel:
    def __init__(self, Filter, bias, stride=1, padding=0):
        self.W = Filter
        self.b = bias
        self.stride = stride
        self.pad = padding

        self.FH, self.FW = self.W.shape
        self.W = self.W.reshape([1, self.FH*self.FW]) # Filterのreshape

    def conv2D(self, x):
        OH, OW = self.out_size(x)
        print(OH, OW)
    
        x   = self.im2col_stride(x, OH, OW) # xのim2col化
        Out = (self.W @ x) + self.b # 計算
        print(self.W)
        print(x)

        return Out.reshape([OH, OW])

    # im2colのためのreshpeは，自作が必要（通常のreshapeでは置換できない）
    # ref: https://qiita.com/kuroitu/items/35d7b5a4bde470f69570#im2colの動作と初期の実装
    # こうすると軽量化される，先達あらまほしきことなり．：_im2colに対して，**軸が反転してると思えば良い**
    def im2col(self, im, OH, OW):
        col = np.empty([self.FH, self.FW, OH, OW])
        for ih in range(self.FH):
            for iw in range(self.FW):
                col[ih, iw, :, :] = im[ih:ih+OH, iw:iw+OW]
        return col.reshape(self.FH*self.FW, OH*OW)
    # ストライド版
    def im2col_stride(self, im, OH, OW):
        im_pad = np.pad(im, [(self.pad, self.pad), (self.pad, self.pad)], 'constant')
        col = np.empty([self.FH, self.FW, OH, OW])
        for ih in range(self.FH):
            ih_max = ih + self.stride*OH
            for iw in range(self.FW):
                iw_max = iw + self.stride*OW
                col[ih,iw,:,:] = im_pad[ih:ih_max:self.stride, iw:iw_max:self.stride]
        return col.reshape(self.FH*self.FW, OH*OW)

    def out_size(self,x):
        H,   W = x.shape

        OH = (H + (2*self.pad) - self.FH)//self.stride + 1
        OW = (W + (2*self.pad) - self.FW)//self.stride + 1
        
        return OH, OW

if __name__ == '__main__':
    F = np.array([[2, 0],
                  [0, 2]])
    Filter = FilterKernel(F, 0, 2) # ストライドなし:stride=1，1マス飛ばす:stride=2，2マス飛ばす:stride=3

    x = np.array([[1, 2, 3, 0],
                  [0, 1, 2, 3],
                  [3, 0, 1, 2],
                  [2, 3, 0, 1]])
    out = Filter.conv2D(x)
    print(out)