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

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1 ,padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.linear1 = nn.Linear(in_features=5*5*256, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 5*5*256)
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.linear3(x)
        return x


def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])

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
    net = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train
    bestNet = AlexNet().to(device)
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
