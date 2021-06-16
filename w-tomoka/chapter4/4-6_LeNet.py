import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0)
    self.fc1 = nn.Linear(5*5*16, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
    self.pool = nn.AvgPool2d(2, stride=2)

  def forward(self, x):
    x = F.sigmoid(self.conv1(x))
    x = self.pool(x)
    x = F.sigmoid(self.conv2(x))
    x = self.pool(x)
    x = x.view(-1, 5*5*16)
    x = F.sigmoid(self.fc1(x))
    x = F.sigmoid(self.fc2(x))
    x = self.fc3(x)
    return x

#dataset
BATCH_SIZE = 128
transform = torchvision.transforms.ToTensor()
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

#学習
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = LeNet()
net = net.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accs = []
test_losses = []
test_accs = []

epochs = 150
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
