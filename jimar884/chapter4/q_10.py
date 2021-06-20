import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import select
import torch
from torch._C import device
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.optim import optimizer
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

batch_size = 128
num_class = 10

class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3r, n3x3, n5x5r, n5x5, npool):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n1x1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n1x1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n3x3r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n3x3r),
            nn.ReLU(),
            nn.Conv2d(in_channels=n3x3r, out_channels=n3x3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=n5x5r, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n5x5r),
            nn.ReLU(),
            nn.Conv2d(in_channels=n5x5r, out_channels=n5x5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=npool, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(npool),
            nn.ReLU()
        )

    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.preInception = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.gap = nn.AvgPool2d(7, 1, 0)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(1024, 10)


    def forward(self, x):
        x = self.preInception(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool1(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool2(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.gap(x)
        x = x.view(-1, 1024)
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
    net = GoogLeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # train
    bestNet = GoogLeNet().to(device)
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
