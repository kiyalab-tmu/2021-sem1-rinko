import numpy as np


class FilterLayer(object):
    def __init__(self, stride, pad):
        self.stride = stride
        self.pad = pad

    # from: https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/util.py
    def im2col(self, input_data, filter_h, filter_w):
        """
        Parameters
        ----------
        input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
        filter_h : フィルターの高さ
        filter_w : フィルターの幅
        stride : ストライド
        pad : パディング
        Returns
        -------
        col : 2次元配列
        """
        N, C, H, W = input_data.shape
        out_h = (H + 2 * self.pad - filter_h) // self.stride + 1
        out_w = (W + 2 * self.pad - filter_w) // self.stride + 1

        img = np.pad(
            input_data,
            [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)],
            "constant",
        )
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + self.stride * out_h
            for x in range(filter_w):
                x_max = x + self.stride * out_w
                col[:, :, y, x, :, :] = img[
                    :, :, y : y_max : self.stride, x : x_max : self.stride
                ]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col


class Convolution(FilterLayer):
    def __init__(self, W, b, stride=1, pad=0):
        super().__init__(stride, pad)
        self.W = W
        self.b = b

    def __call__(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = self.im2col(x, FH, FW)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out


class MaxPooling(FilterLayer):
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        super().__init__(stride, pad)
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def __call__(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = self.im2col(x, self.pool_h, self.pool_w)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out


class AvgPooling(FilterLayer):
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        super().__init__(stride, pad)
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def __call__(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = self.im2col(x, self.pool_h, self.pool_w)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        out = np.mean(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out
