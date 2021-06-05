import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import cv2

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    H, W= input_data.shape
    N = 1
    C = 1
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N,filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:,y, x, :, :] = img[y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 3, 4, 1, 2).reshape(N*out_h*out_w, -1)
    return col


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        #(逆伝播時に使用する)中間データを初期化
        self.x = None #入力データ
        self.col = None #2次元配列に展開した入力データ
        self.col_W = None #2次元配列に展開したフィルター（重み）

        #勾配に関する変数を初期化
        self.dW = None #フィルター（重み）に関する勾配
        self.db = None #バイアスに関する勾配

    #順伝播メソッドの定義
    def forward(self, x):
        #各データに関するサイズを取得
        FH, FW = self.W.shape #フィルター
        FN = 1
        H, W= x.shape #入力データ
        N = 1
        C = 1
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride) #出力データ
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        #各データを2次元配列に展開
        col = im2col(x, FH, FW, self.stride, self.pad) #入力データ
        col_W = self.W.reshape(FN, -1).T #フィルター

        #出力の計算
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out


img_ori = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_256x256.png')
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)
img = np.array(img_gray)

#W = np.array([[1, 0, -1],[0, 0, 0],[-1,0,1]])
#W = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
W = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
b = 0
conv = Convolution(W, b, stride=1, pad=0)
Y = conv.forward(img)
print(Y.shape)
Y = Y[0][0][:][:]
print(Y.shape)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title('input')
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('answer')
plt.imshow(Y, cmap='gray')
plt.show()
