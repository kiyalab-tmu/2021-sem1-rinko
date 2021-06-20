#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
show=ToPILImage()
import numpy as np
import matplotlib.pyplot as plt

batchSize=128
 
##load data
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
nb_train = int(0.8 * len(trainset))
nb_valid = int(0.2 * len(trainset))

trainset, validset = torch.utils.data.dataset.random_split(trainset, [nb_train, nb_valid])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=0)
validloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True)
 
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)
 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

####network
class LeNet(nn.Module):
 
    def __init__(self, input_size, output_size):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_size, 6, kernel_size=5, padding=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Tanh(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
            nn.BatchNorm2d(120),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
            nn.Softmax(),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x

    
net = LeNet(1, 10).cuda()
print (net)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.05,momentum=0.9) 

#train
train_losses = []
train_accs = []
test_losses = []
test_accs = []

epochs = 50
for epoch in range(epochs):
    #学習ループ
    running_loss = 0.0
    running_acc = 0.0
    for inputs, labels in trainloader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        running_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        running_acc += torch.sum(pred == labels)
        optimizer.step()

    running_loss /= len(trainloader)
    running_acc = float(running_acc / len(trainloader.dataset))
    train_losses.append(running_loss)
    train_accs.append(running_acc)

    test_loss = 0.0
    test_acc = 0.0
    for inputs, labels in testloader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        output = net(inputs)
        loss = criterion(output, labels)
        test_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        test_acc += torch.sum(pred == labels)
    test_loss /= len(testloader)
    print(test_acc)
    print(len(testset))
    test_acc = float(test_acc / len(testset))
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f'epoch:{epoch}, '
          f'train_loss: {running_loss:.6f}, train_acc: {running_acc:.6f}, '
          f'test_loss: {test_loss:.6f}, test_acc: {test_acc:.6f}')

fig, ax = plt.subplots(2)
ax[0].plot(train_losses, label='train loss')
ax[0].plot(test_losses, label='test loss')
ax[0].legend()
ax[1].plot(train_accs, label='train acc')
ax[1].plot(test_accs, label='test acc')
ax[1].legend()
plt.savefig("q11.jpg")
