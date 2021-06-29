import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#dataset
BATCH_SIZE = 16
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


#Network in Network
class NiN_Net(nn.Module):
  def __init__(self, input_size=3, num_classes=10):
    super(NiN_Net, self).__init__()
    self.num_classes = num_classes

    self.conv11 = nn.Conv2d(input_size, 192,  kernel_size=5, stride=1, padding=2)
    self.conv12 = nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0)
    self.conv13 = nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0)

    self.bn1 = nn.BatchNorm2d(96)

    self.dropout1 = nn.Dropout2d(0.5)

    self.conv21 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
    self.conv22 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
    self.conv23 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)

    self.bn2 = nn.BatchNorm2d(192)

    self.dropout2 = nn.Dropout2d(0.5)

    self.conv31 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
    self.conv32 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
    self.conv33 = nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0)

    self.bn3 = nn.BatchNorm2d(num_classes)

    #self.fc = nn.Linear(10, num_classes)

  def forward(self, x):
    x = F.relu(self.conv11(x))
    x = F.relu(self.conv12(x))
    x = F.relu(self.conv13(x))

    x = F.max_pool2d(self.bn1(x), 3, stride=2, padding=1)
    x = self.dropout1(x)

    x = F.relu(self.conv21(x))
    x = F.relu(self.conv22(x))
    x = F.relu(self.conv23(x))

    x = F.avg_pool2d(self.bn2(x), 3, stride=2, padding=1)
    x = self.dropout2(x)

    x = F.relu(self.conv31(x))
    x = F.relu(self.conv32(x))
    x = F.relu(self.conv33(x))

    x = self.bn3(x)
    x = F.avg_pool2d(x, kernel_size=56, stride=1, padding=0)
    x = x.view(x.size(0), -1)
    #x = F.avg_pool2d(self.bn3(x), 8, stride=1, padding=0)
    #x = x.view(x.size(0), self.num_classes)
    return x


#学習
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

net = NiN_Net(input_size=1, num_classes=10)
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

