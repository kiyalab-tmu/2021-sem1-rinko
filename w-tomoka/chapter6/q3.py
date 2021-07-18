import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import trange


#detaloader
transform = transforms.Compose([transforms.ToTensor(), 
                               transforms.Normalize((0.5,), (0.5,))])
train_dataset = MNIST('./data', train=True, transform=transform, download=True)

batch_size = 128
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class Generator(nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.main = nn.Sequential(
        #fc1
        nn.Linear(input_dim, 128),
        nn.LeakyReLU(0.2, inplace=True),
        #fc2
        nn.Linear(128, 256),
        nn.LeakyReLU(0.2, inplace=True),
        #fc3
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2, inplace=True),
        #fc4
        nn.Linear(512, output_dim),
        nn.Tanh(),
    )

  def forward(self, x):
    x = self.main(x)
    return x

class Discriminator(nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.main = nn.Sequential(
        #fc1
        nn.Linear(input_dim, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        #fc2
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        #fc3
        nn.Linear(256, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),
        #fc4
        nn.Linear(128 ,1),
        nn.Sigmoid(),
        nn.Flatten(),
    )

  def forward(self, x):
    x = self.main(x)
    return x


#GAN
latent_dim = 100 #ノイズの次元数
data_dim = 28*28 #データ(MNIST)の次元数

#学習過程で Generatorが生成する画像を可視化するためのノイズz
fixed_z = torch.randn(100, latent_dim, device=device)

#label
real_label = 1
fake_label = 0

#Generator, Discriminatorを作成
netG = Generator(latent_dim, data_dim).to(device)
netD = Discriminator(data_dim).to(device)

criterion = nn.BCELoss()
lr = 0.0002
G_optimizer = optim.Adam(netG.parameters(), lr=lr)
D_optimizer = optim.Adam(netD.parameters(), lr=lr)


#Discriminatorを学習する関数
def D_train(x):
  netD.zero_grad()
  #(N, H, W) -> (N, H*W)
  x = x.flatten(start_dim=1)

  #損失関数の計算
  # 1: 入力が本物のデータのとき
  y_pred = netD(x)
  y_real = torch.full_like(y_pred, real_label)
  loss_real = criterion(y_pred, y_real)

  # 2: 入力が偽物のデータのとき
  z = torch.randn(x.size(0), latent_dim, device=device)
  y_pred = netD(netG(z))
  y_fake = torch.full_like(y_pred, fake_label)
  loss_fake = criterion(y_pred, y_fake)

  loss = loss_real + loss_fake
  loss.backward()

  D_optimizer.step()

  return float(loss)

#Generatorを学習する関数
def G_train(x):
  netG.zero_grad()

  #損失関数の計算
  z = torch.randn(x.size(0), latent_dim, device=device)
  y_pred = netD(netG(z))
  y = torch.full_like(y_pred, real_label)
  loss = criterion(y_pred, y)

  loss.backward()
  G_optimizer.step()

  return float(loss)


#Generatorで画像を生成する関数
def generate_img(netG, fixed_z):
  with torch.no_grad():
    #画像を生成
    x = netG(fixed_z)

  #(N, C*H*W) -> (N, C, H, W)
  x = x.view(-1, 1, 28, 28).cpu()
  #画像を格子状に並べる
  img = torchvision.utils.make_grid(x, nrow=10, normalize=True, pad_value=1)
  #tensor -> PIL Image
  img = transforms.functional.to_pil_image(img)

  return img


#GANの学習を実行する関数
def train_gan(n_epoch):
  netG.train()
  netD.train()

  history = []
  for epoch in trange(n_epoch, desc="epoch"):
    D_losses, G_losses = [], []
    for x, _ in trainloader:
      x = x.to(device)
      D_losses.append(D_train(x))
      G_losses.append(G_train(x))

    #途中経過を確認するために画像を生成
    img = generate_img(netG, fixed_z)

    #途中経過を記録
    info = {
        "epoch": epoch+1,
        "D_loss": np.mean(D_losses),
        "G_loss": np.mean(G_losses),
        "img": img,
    }
    history.append(info)
  
  history = pd.DataFrame(history)

  return history


def plot_history(history):
  fig, ax = plt.subplots()

  #lossをplot
  ax.set_title("Loss")
  ax.plot(history["epoch"], history["D_loss"], label="Discriminator")
  ax.plot(history["epoch"], history["G_loss"], label="Generator")
  ax.set_xlabel("Epoch")
  ax.legend()

  plt.show()
  
  
  history = train_gan(n_epoch=50)
  plot_history(history)
  
  def create_animation(imgs):
  imgs[0].save(
      "history.gif", save_all=True, append_images=imgs[1:], duration=500, loop=0
  )

create_animation(history["img"])

display(history["img"].iloc[-1])
