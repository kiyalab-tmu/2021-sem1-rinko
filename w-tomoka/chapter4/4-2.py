import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import cv2

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W= input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  #kernel
        self.b = b  #bias
        self.stride = stride
        self.pad = pad

        #勾配に関する変数を初期化
        self.dW = None #フィルター（重み）に関する勾配
        self.db = None #バイアスに関する勾配

    #順伝播メソッドの定義
    def forward(self, x):
        #各データに関するサイズを取得
        FN, C, FH, FW = self.W.shape #フィルター
        N, C, H, W= x.shape #入力データ
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride) #出力データのheight
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride) #出力データのweight

        #各データを2次元配列に展開
        col = im2col(x, FH, FW, self.stride, self.pad) #入力データ
        col_W = self.W.reshape(FN, -1).T #フィルター

        #出力の計算
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out

#画像の用意-------------------------------------------------
#img_ori = io.mread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_256x256.png')
img_ori = io.imread('https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png')
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
img_gray = np.array(img_gray)[:, :, np.newaxis]
input = np.array(img_ori)[np.newaxis].transpose(0, 3, 1, 2) #引数にinputしたい画像
print(input.shape)


#正規分布でカーネルを初期化------------------------------
dim = 1
kernel_size = 3
channel = input.shape[1]
W = np.random.randn(dim, channel, kernel_size, kernel_size)


#Edge detection用カーネル作成-----------------------------
#W = np.array([[1, 0, -1],[0, 0, 0],[-1,0,1]])
#W = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
#W = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
#W = np.array(W)[np.newaxis, np.newaxis]


b = np.zeros(dim)
conv = Convolution(W, b, stride=1, pad=0)
Y = conv.forward(input)
print(Y.shape)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title('input')
plt.imshow(input[0][0], cmap='gray')
plt.subplot(1, 2, 2)
plt.title('answer')
plt.imshow(Y[0][0], cmap='gray')

plt.show()
