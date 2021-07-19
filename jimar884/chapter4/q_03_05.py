import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST


batch_size = 256
num_class = 10


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1 ,padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(32*6*6, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 10)
    
    def foward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


def softmax(x):
    for i in range(len(x)):
        x[i] = torch.exp(x[i]) / torch.sum(torch.exp(x[i]))
    return x


def cross_entropy(pred_y, y):
    tmp = torch.zeros((len(pred_y), num_class))
    for i in range(len(pred_y)):
        tmp[i, y[i].item()] = 1
    ans = torch.zeros(len(pred_y))
    for i in range(len(pred_y)):
        ans = torch.sum(-(tmp[i]*torch.log(pred_y[i])))
    return torch.mean(ans)


def main():
    # donwload data & make DataLoader
    data_path = 'jimar884/chapter4/data'
    train_dataset = FashionMNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = FashionMNIST(data_path, train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # print(train_dataset.data[0].shape)

    net = Net()
    num_epoch = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.03)
    losses = []

    
    for epoch in range(num_epoch):
        losses.append(0)
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = net(images)
                pred_labels = softmax(outputs)
                loss = criterion(images, labels)
                losses[epoch] += loss.item() / len(labels)
            loss.backward()
            optimizer.step()
        print("epoch:%3d, loss:%.4f" % (epoch, losses[epoch]))

    # get accuracy
    sum_loss = .0
    sum_correct = .0
    sum_total = .0
    for i, (images, labels) in enumerate(test_loader):
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(images, labels)
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted==labels).sum().item()
    print("test mean loss={}, accuracy={}".format(
        sum_loss*batch_size/len(test_loader.dataset), float(sum_correct/sum_total)
    ))
    
    # show loss
    print(losses)
    plt.plot(losses)
    plt.show()

if __name__=="__main__":
    main()

