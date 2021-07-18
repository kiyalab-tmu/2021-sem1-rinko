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
                              batch_size=1,
                              shuffle=False)

class Autoencoder(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(784, 200)
        self.l2 = nn.Linear(200, 784)

    def forward(self, x):
        # エンコーダ
        h = self.l1(x)
        # 活性化関数
        h = torch.relu(h)

        # デコーダ
        h = self.l2(h)
        # シグモイド関数で0～1の値域に変換   
        y = torch.sigmoid(h)

        return y

# モデルの定義
model = Autoencoder(device=device).to(device)
# 損失関数の設定
criterion = nn.BCELoss()
# 最適化関数の設定
optimizer = optimizers.Adam(model.parameters())

epochs = 30
train_losses =[]

#train
for epoch in range(epochs):
    train_loss = 0.0
    for (x, _) in train_dataloader:
        x = x.to(device)
        model.train()
       
        preds = model(x)
        loss = criterion(preds, x)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    
    print('Epoch: {}, Loss: {:.3f}'.format(epoch+1,train_loss))