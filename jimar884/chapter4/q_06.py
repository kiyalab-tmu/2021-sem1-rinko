import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
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

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=5*5*16, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.linear3 = nn.Linear(in_features=84, out_features=10)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # # sigmoid ver
        # x = self.pool1(self.sigmoid(self.conv1(x)))
        # x = self.pool2(self.sigmoid(self.conv2(x)))
        # x = x.view(-1, 5*5*16)
        # x = self.sigmoid(self.linear1(x))
        # x = self.sigmoid(self.linear2(x))
        # x = self.linear3(x)

        # relu ver
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 5*5*16)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def main():
    # donwload data & make DataLoader
    data_path = 'jimar884/chapter4/data'
    train_dataset = FashionMNIST(data_path, train=True, download=False, transform=transforms.ToTensor())
    test_dataset = FashionMNIST(data_path, train=False, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # print(type(train_dataset.data))   # torch.Tensor
    # print(train_dataset.data.shape)   # 60000x28x28

    # define Net
    net = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train
    bestNet = LeNet()
    bestAcc = 0
    for epoch in range(20):
        # print("epoch {0} started".format(epoch))
        running_loss = .0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0], data[1]
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # if i==0:
            #     print(loss)
            #     print(loss.item())
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0], data[1]
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
