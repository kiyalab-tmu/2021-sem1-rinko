import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import average, select
from numpy.lib.shape_base import get_array_prepare
import torch
from torch._C import device
from torch.autograd import grad_mode
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

batch_size = 32
num_class = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Bottleneck(nn.Module):
    def __init__(self, in_channels, grow_rate=32):
        super().__init__()
        middle_channels = grow_rate*4
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, grow_rate, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out   # num of chnnale: grow_rate + num of x's chnnels

def dense(in_channels, num_bottleneck, grow_rate=32):
    layers = []
    for i in range(num_bottleneck):
        layers.append(Bottleneck(in_channels, grow_rate))
        in_channels += grow_rate
    return nn.Sequential(*layers)


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 7, 2, 3)   # 224x224x1 -> 112x112x32
        self.pool = nn.MaxPool2d(3, 2, 1)   # 112x112x32 -> 56x56x32
        self.dense1 = dense(32, 6)   # 56x56x32 -> 56x56x(32*6+32)
        self.trans1 = nn.Sequential(
            nn.Conv2d(32*6+32, 32, 1, 1, 0),   # 56x56x(32*6+32) -> 56x56x32
            nn.AvgPool2d(2, 2)   # 56x56x32 -> 28x28x32
        )
        self.dense2 = dense(32, 12)   # 28x28x32 -> 28x28x(32*12+32)
        self.trans2 = nn.Sequential(
            nn.Conv2d(32*12+32, 32, 1, 1, 0),   # 28x28x(32*12+32) -> 28x28x32
            nn.AvgPool2d(2, 2)   # 14x14x32
        )
        self.dense3 = dense(32, 24)   # 14x14x32 -> 14x14x(32*24+32)
        self.trans3 = nn.Sequential(
            nn.Conv2d(32*24+32, 32, 1, 1, 0),   # 14x14x(32*24+32) -> 14x14x32
            nn.AvgPool2d(2, 2)   # 7x7x32
        )
        self.dense4 = dense(32, 16)   # 7x7x32 -> 7x7x(32*16+32)
        self.gap = nn.AvgPool2d(7, 1, 0)   # 7x7x(32*16+32) -> 1x1x(32*16+32)
        self.linear = nn.Linear(32*16+32, 10)

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
        x = x.view(-1, 32*16+32)
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
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = DenseNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # train
    bestNet = DenseNet().to(device)
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
