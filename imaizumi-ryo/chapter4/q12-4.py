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
transform = transforms.Compose([transforms.Resize(96),transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])

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
class Residual(nn.Module):
    def __init__(self,in_channel,num_channel,use_conv1x1=False,strides=1):
        super(Residual,self).__init__()
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm2d(in_channel,eps=1e-3)
        self.conv1=nn.Conv2d(in_channels =in_channel,out_channels=num_channel,kernel_size=3,padding=1,stride=strides)
        self.bn2=nn.BatchNorm2d(num_channel,eps=1e-3)
        self.conv2=nn.Conv2d(in_channels=num_channel,out_channels=num_channel,kernel_size=3,padding=1)
        if use_conv1x1:
            self.conv3=nn.Conv2d(in_channels=in_channel,out_channels=num_channel,kernel_size=1,stride=strides)
        else:
            self.conv3=None
 
 
    def forward(self, x):
        y=self.conv1(self.relu(self.bn1(x)))
        y=self.conv2(self.relu(self.bn2(y)))
        # print (y.shape)
        if self.conv3:
            x=self.conv3(x)
        # print (x.shape)
        z=y+x
        return z

# blk = Residual(3,3,True)
# X = Variable(torch.zeros(4, 3, 96, 96))
# out=blk(X)
 
def ResNet_block(in_channels,num_channels,num_residuals,first_block=False):
    layers=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            layers+=[Residual(in_channels,num_channels,use_conv1x1=True,strides=2)]
        elif i>0 and not first_block:
            layers+=[Residual(num_channels,num_channels)]
        else:
            layers += [Residual(in_channels, num_channels)]
    blk=nn.Sequential(*layers)
    return blk
 
class ResNet(nn.Module):
    def __init__(self,in_channel,num_classes):
        super(ResNet,self).__init__()
        self.block1=nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=7,stride=2,padding=3),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.block2=nn.Sequential(ResNet_block(64,64,2,True),
                                  ResNet_block(64,128,2),
                                  ResNet_block(128,256,2),
                                  ResNet_block(256,512,2))
        self.block3=nn.Sequential(nn.AvgPool2d(kernel_size=3))
        self.Dense=nn.Linear(512,10)
 
 
    def forward(self,x):
        y=self.block1(x)
        y=self.block2(y)
        y=self.block3(y)
        y=y.view(-1,512)
        y=self.Dense(y)
        return y
 
 
net=ResNet(1,10).cuda()
print (net)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.05,momentum=0.9)
 
#train
train_losses = []
train_accs = []
test_losses = []
test_accs = []

epochs = 10
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
plt.savefig("q12.jpg")
