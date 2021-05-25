import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

batch_size = 256
num_class = 10

def relu(x):
    x[x < 0] = 0
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

def whichclass(pred_y):
    ans = torch.zeros(len(pred_y))
    for i in range(len(pred_y)):
        ans[i] = torch.argmax(pred_y[i])
    return ans



class Net_dropout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.2)
        nn.init.normal_(self.linear1.weight, 0.0, 0.01)
        nn.init.normal_(self.linear2.weight, 0.0, 0.01)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


def main():
    # donwload data & make DataLoader
    data_path = 'jimar884/chapter3/data'
    train_dataset = FashionMNIST(data_path, train=True, download=False, transform=transforms.ToTensor())
    test_dataset = FashionMNIST(data_path, train=False, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # initialize the params
    num, heignt, width = train_dataset.data.shape
    num_hiddenunit = 256
    num_class = 10

    # train
    net = Net_dropout(input_size=heignt*width, hidden_size=num_hiddenunit, output_size=10)
    num_epoch = 100
    optimizer = optim.SGD(net.parameters(), lr=0.03)
    losses = []

    for epoch in range(num_epoch):
        losses.append(0)
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, heignt*width)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = net(images)
                pred_labels = softmax(outputs)
                loss = cross_entropy(pred_labels, labels)
                losses[epoch] += loss / len(labels)
            loss.backward()
            optimizer.step()
        print("epoch:%3d, loss:%.4f" % (epoch, losses[epoch]))

    # get accuracy
    correct = 0.0
    count = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.view(-1, heignt*width)
        outputs = net(images)
        pred_labels = softmax(outputs)
        pred_labels = whichclass(pred_labels)
        for j in range(len(pred_labels)):
            if pred_labels[j].int() == labels[j]:
                correct += 1
            count += 1
    acc = correct / count
    print("accuracy:%.4f" % (acc))
    
    # show loss
    plt.plot(losses)
    plt.show()

if __name__=='__main__':
    main()

