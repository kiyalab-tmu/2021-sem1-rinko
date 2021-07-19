import torch
import torchvision
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# 乱数のシード（種）を固定
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# データの取得
root = os.path.join('data', 'MNIST')
transform = transforms.Compose([transforms.ToTensor(),
                                lambda x: x.view(-1)])
mnist_train = \
    torchvision.datasets.MNIST(root=root,
                                      download=True,
                                      train=True,
                                      transform=transform)
mnist_test = \
    torchvision.datasets.MNIST(root=root,
                                      download=True,
                                      train=False,
                                      transform=transform)
train_dataloader = DataLoader(mnist_train,
                              batch_size=100,
                              shuffle=True)
test_dataloader = DataLoader(mnist_test,
                              batch_size=100,
                              shuffle=False)

class VAE(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = Encoder(device=device)
        self.decoder = Decoder(device=device)

    def forward(self, x):
        # エンコーダ
        mean, var = self.encoder(x)
        # 潜在変数の作成
        z = self.reparameterize(mean, var)
        # デコーダ
        y = self.decoder(z)
        #生成画像yと潜在変数zが返り値
        return y, z

    # 潜在変数zの作成
    def reparameterize(self, mean, var):
        # 標準正規分布の作成
        eps = torch.randn(mean.size()).to(self.device)
        # 再パラメータ化トリック
        z = mean + torch.sqrt(var) * eps
        return z

    # 誤差の計算
    def lower_bound(self, x):
        # 平均と分散のベクトルを計算
        mean, var = self.encoder(x)
        # 平均と分散から潜在変数zを作成
        z = self.reparameterize(mean, var)
        # 潜在変数zから生成画像を作成
        y = self.decoder(z)
        # 再構成誤差
        reconst =  - torch.mean(torch.sum(x * torch.log(y)
                                       + (1 - x) * torch.log(1 - y),
                                       dim=1))
        # 正則化
        kl = - 1/2 * torch.mean(torch.sum(1
                                          + torch.log(var)
                                          - mean**2
                                          - var, dim=1))
        # 再構成誤差 + 正則化
        L =  reconst + kl

        return L
    
class Encoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(784, 200)
        self.l_mean = nn.Linear(200, 10)
        self.l_var = nn.Linear(200, 10)

    def forward(self, x):
        # 784次元から200次元
        h = self.l1(x)
        # 活性化関数
        h = torch.relu(h)
        # 200次元から10次元の平均
        mean = self.l_mean(h)
        # 200次元から10次元の分散
        var = self.l_var(h)
        # 活性化関数softplus
        var = F.softplus(var)

        return mean, var

class Decoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(10, 200)
        self.out = nn.Linear(200, 784)

    def forward(self, x):
        # 10次元から200次元
        h = self.l1(x)
        # 活性化関数
        h = torch.relu(h)
        # 200次元から784次元
        h = self.out(h)
        # シグモイド関数
        y = torch.sigmoid(h)

        return y
    
# モデルの設定
model = VAE(device=device).to(device)
# 損失関数の設定
criterion = model.lower_bound
# 最適化関数の設定
optimizer = optimizers.Adam(model.parameters())

epochs = 20
train_losses =[]

for epoch in range(epochs):
    train_loss = 0.
    # バッチサイズのループ
    for (x, _) in train_dataloader:
        x = x.to(device)       
        model.train()

        loss = criterion(x)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    print('Epoch: {}, Loss: {:.3f}'.format(epoch+1,train_loss))

#plot(loss)
plt.plot(range(1, epochs+1),train_losses,label="train")
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./q2_loss.png')
plt.close()    

#plot(image)
# ノイズの作成数
batch_size=8
z = torch.randn(batch_size, 10, device = device)
model.eval()

images = model.decoder(z)
images = images.view(-1, 28, 28)
images = images.squeeze().detach().cpu().numpy()

for i, image in enumerate(images):
    plt.subplot(2, 4, i+1)
    plt.imshow(image, cmap='binary_r')
    plt.axis('off')
plt.tight_layout()
plt.savefig("Q2.png")
plt.close()

fig = plt.figure(figsize=(10, 3))
model.eval()
for x, t in test_dataloader:
    # 本物画像
    for i, im in enumerate(x.view(-1, 28, 28).detach().numpy()[:10]):
      ax = fig.add_subplot(3, 10, i+1, xticks=[], yticks=[])
      ax.imshow(im, 'gray')
    x = x.to(device)
    # 本物画像から生成画像
    y, z = model(x)
    y = y.view(-1, 28, 28)
    for i, im in enumerate(y.cpu().detach().numpy()[:10]):
      ax = fig.add_subplot(3, 10, i+11, xticks=[], yticks=[])
      ax.imshow(im, 'gray')
    # 1つ目の画像と2つ目の画像の潜在変数を連続的に変化
    z1to0 = torch.cat([z[1] * (i * 0.1) + z[0] * ((9 - i) * 0.1) for i in range(10)]).reshape(10,10)
    y2 = model.decoder(z1to0).view(-1, 28, 28)
    for i, im in enumerate(y2.cpu().detach().numpy()):
      ax = fig.add_subplot(3, 10, i+21, xticks=[], yticks=[])
      ax.imshow(im, 'gray')
    break

plt.savefig("Q2_images.png")
plt.close()