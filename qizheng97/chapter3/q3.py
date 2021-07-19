import torch
import numpy as np
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.output=nn.Linear(256,10)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.hidden(x))
        x=self.output(x)
        x = F.softmax(x, dim=1)
        return x

batch_size=256
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=0,std=0.01)
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.FashionMNIST("dataset/", download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.FashionMNIST("dataset/", download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

image, label = next(iter(trainloader))

model = Classifier()
device = torch.device('cuda:0')
model.to(device)
#criterion = nn.CrossEntropyLoss()
#
criterion = nn.CrossEntropyLoss()

# 优化方法为Adam梯度下降方法，学习率为0.003
optimizer = optim.SGD(model.parameters(), lr=0.003)


epoches = 50

print("train start")

train_losses, test_losses = [], []
for i in range(epoches):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        images=images.to(device)
        labels=labels.to(device)
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    test_loss = 0
    accuracy = 0

    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        test_loss += criterion(out, labels)
        values,index=torch.topk(input=out,k=1,dim=1)
        result=(index==labels.view(*index.shape))

        accuracy += torch.mean(result.type(torch.FloatTensor))



    train_losses.append(running_loss / len(trainloader))
    test_losses.append(test_loss / len(testloader))

    print("训练集学习次数: {}/{}.. ".format(i, epoches),
          "训练误差: {:.3f}.. ".format(running_loss / len(trainloader)),
          "测试误差: {:.3f}.. ".format(test_loss / len(testloader)),
          "模型分类准确率: {:.3f}".format(accuracy / len(testloader)))

# imagedemo=image[5].reshape((28,28))
# imagedemolabel=label[3]
# print(imagedemolabel)
# plt.imshow(imagedemo)
# plt.show()
