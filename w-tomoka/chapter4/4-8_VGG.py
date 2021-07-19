import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class VGG11(nn.Module):
  def __init__(self, input_size=3, num_classes):
    super(VGG11, self).__init__()

    self.block1 = nn.Sequential(
        nn.Conv2d(input_size, 64, kernel_size=3, padding=1), 
        nn.ReLU(inplace=True), 
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.block2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, padding=1), 
        nn.ReLU(inplace=True), 
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.block3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, padding=1), 
        nn.ReLU(inplace=True), 
        nn.Conv2d(256, 256, kernel_size=3, padding=1), 
        nn.ReLU(inplace=True), 
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.block4 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, padding=1), 
        nn.ReLU(inplace=True), 
        nn.Conv2d(512, 512, kernel_size=3, padding=1), 
        nn.ReLU(inplace=True), 
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.block5 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, padding=1), 
        nn.ReLU(inplace=True), 
        nn.Conv2d(512, 512, kernel_size=3, padding=1), 
        nn.ReLU(inplace=True), 
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.classifier = nn.Sequential(
        nn.Linear(512*7*7, 4096), 
        nn.ReLU(inplace=True), 
        nn.Dropout(), 
        nn.Linear(4096, 4096), 
        nn.ReLU(inplace=True), 
        nn.Dropout(), 
        nn.Linear(4096, num_classes), 
    )

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)

    x = x.view(-1, 5*5*256)
    x = self.classifier(x)
    return x

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

#学習
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = LeNet(1,10)
net = net.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accs = []
test_losses = []
test_accs = []

epochs = 30
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
