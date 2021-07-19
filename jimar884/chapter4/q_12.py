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
from torch.optim import optimizer
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

batch_size = 256
num_class = 10

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
            # nn.ReLU()
        )
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.layer(x)
        out += shortcut
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 64, 7, 2, 3)   # 224x224x1 -> 112x112x64
        self.layer2 = nn.MaxPool2d(3, 2, 1)   # 112x112x64 -> 56x56x64
        self.layer3_4 = Block(64, 64)
        self.layer5_6 = Block(64,64)
        self.layer7_8 = Block(64, 128, 2)   # 56x56x64 -> 28x28x128
        self.layer9_10 = Block(128, 128)
        self.layer11_12 = Block(128, 256, 2)   # 28x28x128 -> 14x14x256
        self.layer13_14 = Block(256, 256)
        self.layer15_16 = Block(256, 512, 2)   # 14x14x256 -> 7x7x512
        self.layer17_18 = Block(512, 512)
        self.gap = nn.AvgPool2d(7, 1, 0)   # 7x7x512 -> 1x1x512
        self.linear = nn.Linear(512, 10)   # 10 is the number of classes

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3_4(x)
        x = self.layer5_6(x)
        x = self.layer7_8(x)
        x = self.layer9_10(x)
        x = self.layer11_12(x)
        x = self.layer13_14(x)
        x = self.layer15_16(x)
        x = self.layer17_18(x)
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
            # if i==0:
            #     print(loss)
            #     print(loss.item())
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
