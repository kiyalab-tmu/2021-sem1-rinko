"""
参照：https://pystyle.info/pytorch-gan/#outline__1
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from tqdm.notebook import trange

device = "cuda:0" if (torch.cuda.is_available()) else "cpu"
print(device)

# Transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

# Dataset
download_dir = "/data"
dataset = datasets.MNIST(download_dir, train=True, transform=transform, download=True)

# DataLoader
batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


#Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.main = nn.Sequential(
            # fc1
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # fc2
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # fc3
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            # fc4
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.main(x)

#Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.main = nn.Sequential(
            # fc1
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # fc2
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # fc3
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # fc4
            nn.Linear(128, 1),
            nn.Sigmoid(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.main(x)
    
def D_train(x):
    D.zero_grad()
    # (N, H, W) -> (N, H * W) に形状を変換
    x = x.flatten(start_dim=1)

    #real
    y_pred = D(x)
    y_real = torch.full_like(y_pred, real_label)
    loss_real = criterion(y_pred, y_real)

    #fake
    z = torch.randn(x.size(0), latent_dim, device=device)
    y_pred = D(G(z))
    y_fake = torch.full_like(y_pred, fake_label)
    loss_fake = criterion(y_pred, y_fake)

    loss = loss_real + loss_fake
    loss.backward()
    D_optimizer.step()

    return float(loss)

def G_train(x):
    G.zero_grad()

    
    z = torch.randn(x.size(0), latent_dim, device=device)
    y_pred = D(G(z))
    y = torch.full_like(y_pred, real_label)
    loss = criterion(y_pred, y)

    loss.backward()
    G_optimizer.step()

    return float(loss)

def generate_img(G, fixed_z):
    with torch.no_grad():
        # 画像生成
        x = G(fixed_z)

    # (N, C * H * W) -> (N, C, H, W) に形状を変換
    x = x.view(-1, 1, 28, 28).cpu()
    # 画像を保存。
    img = torchvision.utils.make_grid(x, nrow=10, normalize=True, pad_value=1)
    img = transforms.functional.to_pil_image(img)

    return img

def train_gan(n_epoch,G,D):
    G.train()
    D.train()

    history = []
    for epoch in trange(n_epoch, desc="epoch"):

        D_losses, G_losses = [], []
        for x, _ in dataloader:
            x = x.to(device)
            D_losses.append(D_train(x))
            G_losses.append(G_train(x))

        # 途中経過を確認するために画像を生成する。
        img = generate_img(G, fixed_z)

        # 途中経過を記録する。
        info = {
            "epoch": epoch + 1,
            "D_loss": np.mean(D_losses),
            "G_loss": np.mean(G_losses),
            "img": img,
        }
        history.append(info)

    history = pd.DataFrame(history)

    return history

def plot_history(history):
    fig, ax = plt.subplots()

    # 損失の推移を描画する。
    ax.set_title("Loss")
    ax.plot(history["epoch"], history["D_loss"], label="Discriminator")
    ax.plot(history["epoch"], history["G_loss"], label="Generator")
    ax.set_xlabel("Epoch")
    ax.legend()

    plt.savefig("./Q3_loss.png")
    plt.close()
    
if __name__ == '__main__':
    latent_dim = 100  #ノイズの次元数
    data_dim = 28 * 28  #データの次元数

    #生成用ノイズ
    fixed_z = torch.randn(100, latent_dim, device=device)

    #label
    real_label = 1
    fake_label = 0


    G = Generator(latent_dim, data_dim).to(device)
    D = Discriminator(data_dim).to(device)

    criterion = nn.BCELoss()

    G_optimizer = optim.Adam(G.parameters(), lr=0.0002)
    D_optimizer = optim.Adam(D.parameters(), lr=0.0002)

    history = train_gan(n_epoch=50,G=G,D=D)
    plot_history(history)
    #最後の画像を表示
    display(history["img"].iloc[-1])