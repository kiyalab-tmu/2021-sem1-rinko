import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import select
from numpy.lib.shape_base import get_array_prepare
import torch
from torch._C import device
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AvgPool2d
from torch.optim import optimizer
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

batch_size = 256
num_class = 10

class Dense(nn.Module):
    def __init__(self, in_channels, out_channels, num, stride=1):
        super().__init__()
        self.layers = [nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            # nn.BatchNorm2d(out_channels)
            # nn.ReLU()
        )] * num
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.num = num

    def forward(self, x):
        last_maps = 0
        map = x
        for layer in self.layers:
            last_maps += map
            map = layer(self.relu(self.bn(last_maps)))
        out = self.relu(self.bn(map))
        return out


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, 7, 2, 3)   # 224x224x1 -> 112x112x64
        self.pool = nn.MaxPool2d(3,2, 1)   # 112x112x64 -> 56x56x64
        self.dense1 = Dense(64, 64, 6)   # 56x56x64
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, 0),   # 56x56x64 -> 56x56x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0)   # 56x56x128 -> 28x28x128
        )
        self.dense2 = Dense(128, 128, 12)   # 28x28x128
        self.trans2 = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1, 0),   # 28x28x128 -> 28x28x256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0)   # 28x28x256 -> 14x14x256
        )
        self.dense3 = Dense(256, 256, 24)   # 14x14x256
        self.trans3 = nn.Sequential(
            nn.Conv2d(256, 512, 1, 1, 0),   # 14x14x256 -> 14x14x512
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0)   # 14x14x512 -> 7x7x512
        )
        self.dense4 = Dense(512, 512, 16)   # 7x7x512
        self.gap = nn.AvgPool2d(7, 1, 0)   # 7x7x512 -> 1x1x512
        self.linear = nn.Linear(512, 10)   # 10 is the number of classes

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        x = self.gap(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])

    # donwload data & make DataLoader
    data_path = 'data'
    train_dataset = FashionMNIST(data_path, train=True, download=False, transform=transform)
    test_dataset = FashionMNIST(data_path, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # print(type(train_dataset.data))   # torch.Tensor
    # print(train_dataset.data.shape)   # 60000x28x28

    # define Net
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # train
    bestNet = ResNet18().to(device)
    bestAcc = 0
    losses = []
    for epoch in range(20):
        # print("epoch {0} started".format(epoch))
        epoch_loss = .0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss = epoch_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
        current_accuracy = correct / total
        print("acc in epoch{0} : {1}".format(epoch+1, current_accuracy))
        if current_accuracy > bestAcc:
            bestAcc = current_accuracy
            bestNet = copy.deepcopy(net)
    net = bestNet
    print(bestAcc)


if __name__=='__main__':
    main()
