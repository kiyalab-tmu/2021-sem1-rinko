import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math

#dataset
BATCH_SIZE = 8
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


#Dense Net

class DenseLayer(nn.Module):
  def __init__(self, n, growth_rate):
    super(DenseLayer, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=growth_rate*n, 
                           out_channels=growth_rate*4, 
                           kernel_size=1, stride=1)
    self.conv2 = nn.Conv2d(in_channels=growth_rate*4, 
                           out_channels=growth_rate, 
                           kernel_size=3, stride=1, padding=1)
    
    self.bn1 = nn.BatchNorm2d(growth_rate*n)
    self.bn2 = nn.BatchNorm2d(growth_rate*4)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv1(x)

    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv2(x)

    return x

class DenseBlock(nn.Module):
  def __init__(self, growth_rate, num_layers):
    super(DenseBlock, self).__init__()
    self.layers = nn.ModuleList([DenseLayer(n+1, growth_rate) for n in range(num_layers)])

  def forward(self, x):
    for layer in self.layers:
      x = torch.cat((x, layer(x)), 1)
    return x

class TransitionBlock(nn.Module):
  def __init__(self, growth_rate, theta, n):
    super(TransitionBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels=growth_rate*(n+1), 
                          out_channels=growth_rate, 
                          kernel_size=1, stride=1, padding=0)
    self.avg_pool = nn.AvgPool2d(kernel_size=2, 
                                 stride=math.ceil(1/theta))
    
    self.bn = nn.BatchNorm2d(growth_rate*(n+1))
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x)
    x = self.avg_pool(x)

    return x

class DenseNet(nn.Module):
  def __init__(self, growth_rate=16, theta=0.5, num_layers=[6, 12, 24, 16], 
               in_channels=3, num_classes=10):
    super(DenseNet, self).__init__()
    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=growth_rate,
                          kernel_size=7, stride=2, padding=3)
    self.bn = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.block1 = DenseBlock(growth_rate, num_layers[0])
    self.transitionBlock1 = TransitionBlock(growth_rate, theta, num_layers[0])
    self.block2 = DenseBlock(growth_rate, num_layers[1])
    self.transitionBlock2 = TransitionBlock(growth_rate, theta, num_layers[1])
    self.block3 = DenseBlock(growth_rate, num_layers[2])
    self.transitionBlock3 = TransitionBlock(growth_rate, theta, num_layers[2])
    self.block4 = DenseBlock(growth_rate, num_layers[3])

    self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
    self.fc = nn.Linear(growth_rate * (num_layers[-1] + 1), num_classes)
  
  def forward(self, x):
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv(x)
    x = self.max_pool(x)

    x = self.block1(x)
    x = self.transitionBlock1(x)

    x = self.block2(x)
    x = self.transitionBlock2(x)

    x = self.block3(x)
    x = self.transitionBlock3(x)

    x = self.block4(x)

    x = self.avg_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x



#学習
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

net = DenseNet(in_channels=1, num_classes=10)
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
plt.savefig('/home/wtomoka/rinko/figure.jpg')

