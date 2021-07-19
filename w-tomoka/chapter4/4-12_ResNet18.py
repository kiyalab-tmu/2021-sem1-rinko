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


#ResNet18

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                   padding=dilation, groups=groups, bias=True, dilation=dilation)
  
def conv1x1(in_channels, out_channels, stride=1):
  return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):

  def __init__(self, in_channels, out_channels, stride=1, is_first_resblock=False):
    super(BasicBlock, self).__init__()

    self.is_dim_changed = (in_channels != out_channels)
    if self.is_dim_changed:
      if is_first_resblock:
        stride = 1
      else:
        stride = 2
      self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
    else:
      stride = 1

    self.conv1 = conv3x3(in_channels, out_channels, stride)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(out_channels, out_channels)
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    identity_x = x #hold input for shortcut connection

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.is_dim_changed:
      identity_x = self.shortcut(x)

    out += identity_x  # shortcut connection
    out = self.relu(out)
    return out


class ResNet18(nn.Module):

  def __init__(self, input_size=3, num_classes=100):
    super(ResNet18, self).__init__()
    self.conv1 = nn.Conv2d(input_size, 64, kernel_size=7, stride=2, padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = BasicBlock(in_channels=64, out_channels=64, is_first_resblock=True)
    self.layer2 = BasicBlock(in_channels=64, out_channels=64)
    self.layer3 = BasicBlock(in_channels=64, out_channels=128)
    self.layer4 = BasicBlock(in_channels=128, out_channels=128)
    self.layer5 = BasicBlock(in_channels=128, out_channels=256)
    self.layer6 = BasicBlock(in_channels=256, out_channels=256)
    self.layer7 = BasicBlock(in_channels=256, out_channels=512)
    self.layer8 = BasicBlock(in_channels=512, out_channels=512)
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.maxpool(out)

    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = self.layer6(out)
    out = self.layer7(out)
    out = self.layer8(out)

    out = self.avg_pool(out)
    out = out.view(out.size(0), -1)
    out = self.fc(out)

    return out
  
  
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
