import torch
import numpy as np
import random
from matplotlib import pyplot as plt
 
def make_dataset(N=1000, w=np.array([2, -3.4]), b=4.2):
    noise = np.random.normal(
        loc = 0,
        scale = 0.01,
        size = N
    )
    w = np.transpose(w)
    X = np.empty((N, 2))
    Y = np.empty((N, 1))
    for i in range(N):
        x = np.random.rand(2)
        y = np.dot(x,w) + b + noise[i]
        X[i,0] = x[0]
        X[i,1] = x[1]
        Y[i] = y
    
    return X, Y

X, y = make_dataset()

# サンプルデータをTensorにする
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()


batch_size = 500
data_size = 1000

def linear_regression(dimension, epoch, lr, x, y):
    net = torch.nn.Linear(in_features=dimension, out_features=1)  # ネットワークに線形結合モデルを設定
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)          # 最適化にSGDを設定
    E = torch.nn.MSELoss()                                        # 損失関数にMSEを設定

    w = random.normalvariate(0, 0.01)
    net.weight.data.fill_(w)
    net.bias.data.fill_(0)
 

    # 学習ループ
    losses = []
    for i in range(epoch):
        batch_choice = np.random.choice(1000, batch_size)
        x_batch = x[batch_choice]
        y_batch = y[batch_choice]

        optimizer.zero_grad()                                       # 勾配情報を0に初期化
        y_pred = net(x_batch)                                       # 予測
        loss = E(y_pred.reshape(y_batch.shape), y_batch)            # 損失を計算(shapeを揃える)
        loss.backward()                                             # 勾配の計算
        optimizer.step()                                            # 勾配の更新
        losses.append(loss.item())                                  # 損失値の蓄積
        print(list(net.parameters()))
        print('loss', loss.item())
 
    # 回帰係数を取得して回帰直線を作成
    w0 = net.weight.data.numpy()[0, 0]
    w1 = net.weight.data.numpy()[0, 1]
    x_new = np.linspace(np.min(x.T[1].data.numpy()), np.max(x.T[1].data.numpy()), len(x))
    y_curve = w0 + w1 * x_new
 
    # グラフ描画
    plot(x.T[1], y, x_new, y_curve, losses)
    return net, losses
 
def plot(x, y, x_new, y_pred, losses):
    # ここからグラフ描画-------------------------------------------------
    # フォントの種類とサイズを設定する。
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'
 
    # 目盛を内側にする。
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
 
    # グラフの上下左右に目盛線を付ける。
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax2 = fig.add_subplot(122)
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')
 
    # 軸のラベルを設定する。
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
 
    # スケール設定
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 30)
    ax2.set_xlim(-100, 4000)
    ax2.set_ylim(-0.3, 10)
    #ax2.set_yscale('log')
 
    # データプロット
    ax1.scatter(x, y, label='dataset')
    ax1.plot(x_new, y_pred, color='red', label='PyTorch result')
    ax2.plot(np.arange(0, len(losses), 1), losses)
    #ax2.text(1000, 8, 'Loss=' + str(round(losses[len(losses)-1], 5)), fontsize=16)
    #ax2.text(1000, 9, 'Epoch=' + str(round(len(losses), 1)), fontsize=16)
 
    # グラフを表示する。
    ax1.legend()
    fig.tight_layout()
    plt.show()
    plt.close()
    # -------------------------------------------------------------------
 
# 線形回帰を実行
net, losses = linear_regression(dimension=2, epoch=4000, lr=0.01, x=X, y=y)
