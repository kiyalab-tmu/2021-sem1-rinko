import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#dataset
BATCH_SIZE = 32
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor()])
#60,000
fashion_mnist_data = torchvision.datasets.FashionMNIST(
    root = './fashion-mnist', 
    train = True, 
    download = True, 
    transform = transform)
data_loader = torch.utils.data.DataLoader(
    dataset=fashion_mnist_data,
    batch_size=BATCH_SIZE,
    shuffle=True)

#10,000
fashion_mnist_data_test = torchvision.datasets.FashionMNIST(
    root = './fashion-mnist', 
    train = False, 
    download = True, 
    transform = transform)
data_loader_test = torch.utils.data.DataLoader(
    dataset=fashion_mnist_data_test,
    batch_size=BATCH_SIZE,
    shuffle=False)


#GoogLeNet
class Inception(nn.Module):
  def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
    super(Inception, self).__init__()

    #1x1conv branch
    self.b1 = nn.Sequential(
        nn.Conv2d(input_channels, n1x1, kernel_size=1),
        nn.BatchNorm2d(n1x1),
        nn.ReLU(inplace=True)
    )

    #1x1conv -> 3x3conv branch
    self.b2 = nn.Sequential(
        nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
        nn.BatchNorm2d(n3x3_reduce),
        nn.ReLU(inplace=True),
        nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
        nn.BatchNorm2d(n3x3),
        nn.ReLU(inplace=True)
    )

    #1x1conv -> 5x5conv branch
    self.b3 = nn.Sequential(
        nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
        nn.BatchNorm2d(n5x5_reduce),
        nn.ReLU(inplace=True),
        nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
        nn.BatchNorm2d(n5x5, n5x5),
        nn.ReLU(inplace=True),
        nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
        nn.BatchNorm2d(n5x5),
        nn.ReLU(inplace=True)   
    )

    #3x3pooling -> 1x1conv
    self.b4 = nn.Sequential(
        nn.MaxPool2d(3, stride=1, padding=1),
        nn.Conv2d(input_channels, pool_proj, kernel_size=1),
        nn.BatchNorm2d(pool_proj),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogLeNet(nn.Module):
  def __init__(self, input_size=3, num_classes=100):
    super(GoogLeNet, self).__init__()
    self.prelayer = nn.Sequential(
        nn.Conv2d(input_size, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(192),
        nn.ReLU(inplace=True),
        )
    
    self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
    self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

    self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

    self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
    self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
    self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
    self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
    self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
 
    self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
    self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
 
    #input feature size: 8*8*1024
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout2d(p=0.4)
    self.linear = nn.Linear(1024, num_classes)

  def forward(self, x):
    x = self.prelayer(x)
    x = self.maxpool(x)
    x = self.a3(x)
    x = self.b3(x)

    x = self.maxpool(x)
 
    x = self.a4(x)
    x = self.b4(x)
    x = self.c4(x)
    x = self.d4(x)
    x = self.e4(x)
 
    x = self.maxpool(x)
 
    x = self.a5(x)
    x = self.b5(x)
 
    x = self.avgpool(x)
    x = self.dropout(x)
    x = x.view(x.size()[0], -1)
    x = self.linear(x)

    return x



#学習
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

net = ResNet18(input_size=1, num_classes=10)
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accs = []
test_losses = []
test_accs = []

epochs = 10
for epoch in range(epochs):
  #学習ループ
  running_loss = 0.0
  running_acc = 0.0
  for x, labels in data_loader:
    x = x.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, labels)
    loss.backward()
    running_loss += loss.item()
    pred = torch.argmax(output, dim=1)
    running_acc += (pred == labels).sum().item()
    optimizer.step()

  running_loss /= len(data_loader)
  running_acc = float(running_acc / len(fashion_mnist_data))
  train_losses.append(running_loss)
  train_accs.append(running_acc)

  test_loss = 0.0
  test_acc = 0.0
  for x, labels in data_loader_test:
    x = x.to(device)
    labels = labels.to(device)
    output = net(x)
    loss = criterion(output, labels)
    test_loss += loss.item()
    pred = torch.argmax(output, dim=1)
    test_acc += (pred == labels).sum().item()
  test_loss /= len(data_loader_test)
  test_acc = float(test_acc / len(fashion_mnist_data_test))
  test_losses.append(test_loss)
  test_accs.append(test_acc)

  print(f'epoch:{epoch}, '
        f'train_loss: {running_loss:.6f}, train_acc: {running_acc:.6f}, '
        f'test_loss: {test_loss:.6f}, test_acc{test_acc:.6f}')
  

fig, ax = plt.subplots(2)
ax[0].plot(train_losses, label='train loss')
ax[0].plot(test_losses, label='test loss')
ax[0].legend()
ax[1].plot(train_accs, label='train acc')
ax[1].plot(test_accs, label='test acc')
ax[1].legend()
plt.show()

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

plt.plot(train_accs, label='train acc')
plt.plot(test_accs, label='test acc')
plt.legend()
plt.show()
