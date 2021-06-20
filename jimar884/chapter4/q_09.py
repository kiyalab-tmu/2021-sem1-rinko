import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch._C import device
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.optim import optimizer
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

batch_size = 256
num_class = 10

class NiN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(256, 10, 1, 1, 0),   # 10 is the number of classes
            nn.ReLU()
        )
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.gap = nn.AvgPool2d(7, 1, 0)   # global average pooling: 7 is the size of last layers(7x7x256)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.mlp1(x)   # 28x28x1 -> 28x28x64
        x = self.dropout(x)
        x = self.max_pool(x)   # 28x28x64 -> 14x14x64
        x = self.mlp2(x)   # 14x14x64 -> 14x14x128
        x = self.dropout(x)
        x = self.max_pool(x)   # 14x14x128 -> 7x7x128
        x = self.mlp3(x)   # 7x7x128 -> 7x7x10
        x = self.gap(x)   # 7x7x10 -> 1x1x10
        x = x.view(-1, 1*1*10)
        return x


def main():
    transform = transforms.Compose([transforms.ToTensor()])

    # donwload data & make DataLoader
    data_path = 'jimar884/chapter4/data'
    train_dataset = FashionMNIST(data_path, train=True, download=False, transform=transform)
    test_dataset = FashionMNIST(data_path, train=False, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # print(type(train_dataset.data))   # torch.Tensor
    # print(train_dataset.data.shape)   # 60000x28x28

    # define Net
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = NiN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # train
    bestNet = NiN().to(device)
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
        print("acc in epoch{0} : {1}".format(epoch, current_accuracy))
        if current_accuracy > bestAcc:
            bestAcc = current_accuracy
            bestNet = copy.deepcopy(net)
    net = bestNet


if __name__=='__main__':
    main()
