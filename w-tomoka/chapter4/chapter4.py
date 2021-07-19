import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import cv2




def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def im2col_2d(input_data, filter_h, filter_w, stride=1, pad=0):
    H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(pad, pad), (pad, pad)], 'constant')
    col = np.zeros((out_h,*out_w, filter_h*filter_w))

    for y in range(out_h):
        y_max = y + stride*filter_h
        for x in range(out_w):
            x_max = x + stride*filter_w
            for yy in range(y, y_max):
                for xx in range(x, x_max):
                    col[y*x, yy*xx] = img[yy, xx]
    
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
        FN, C, FH, FW = self.W.shape #フィルター
        N, C, H, W = x.shape #入力データ
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride) #出力データ
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        #各データを2次元配列に展開
        col = im2col(x, FH, FW, self.stride, self.pad) #入力データ
        col_W = self.W.reshape(FN, -1).T #フィルター

        #出力の計算
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        #(逆伝播時に使用する)中間データを保存
        self.x = x
        self.col = col
        self.col_W = col_W

        return out


class Pooling :
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        #逆伝播用の中間データ
        self.x = None #入力データ
        self.arg_max = None #最大値のインデックス

    #順伝播メソッドの定義
    def forward(self, x, method='Max'):
        N, C, H, W = x.shape #入力サイズ
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        #入力データを2次元配列に展開
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        #逆伝播用に最大値のインデックスを保存
        arg_max = np.argmax(col, axis=1)

        #出力データを作成
        if method=='Max':
            out = np.max(col, axis=1)
        elif method=='Average':
            out = np.average(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        #逆伝播用に中間データを保存
        self.x = x #入力データ
        self.arg_max = arg_max #最大値のインデックス

        return out

    #逆伝播メソッドの定義
    #def backward(self, dout):
    #    #入力データを変型
    #    dout = dout.transpose(0,2,3,1)

        #受け皿を作成
    #    pool_size = self.pool_h * self.pool_w #pooling適用領域の要素数
    #    dmax = np.zeros((dout.size, pool_size)) #初期化

        #最大値の要素のみ伝播
    #    dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()

        #4次元配列に変換
    #    dmax = dmax.reshape(dout.shape + (pool_size,))
    #    dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
    #    dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

    #    return dx


class edge_Detection:
    def __init__(self, stride=1, pad=0):
        self.W = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        self.b = 0
        self.stride = stride
        self.pad = pad

    #順伝播メソッドの定義
    def forward(self, image):
        #各データに関するサイズを取得
        FH, FW = self.W.shape #フィルター
        H, W = image.shape #入力データ
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride) #出力データ
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        #各データを2次元配列に展開
        col = im2col_2d(image, FH, FW, stride=1, pad=0) #入力データ
        col_W = self.W.reshape(1, -1).T #フィルター

        #出力の計算
        #!!!!!!ここ多分変！！！！！！！！！！！
        out = np.dot(col, col_W) + self.b
        out = out.reshape(out_h, out_w)
        #out_max = np.max(out)
        #out = int(out / out_max * 255)

        return out


img_ori = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_256x256.png')
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)
img = np.array(img_gray)

#W = np.random.rand(3, 3, 5, 5)
#b = 0
#conv = Convolution(W, b, stride=1, pad=0)
#Y = conv.forward(img)


#pool = Pooling(2,2,stride=2, pad=0)
#Y = pool.forward(X, method='Average')

edge = edge_Detection()
Y = edge.forward(img)

print(Y.shape)

#plt.figure(figsize=(12, 3))
#plt.subplot(1, 2, 1)
#plt.title('input')
#plt.imshow(img_gray)
#plt.subplot(1, 2, 2)
#plt.title('answer')
#plt.imshow(Y)
#plt.show()
