import torch
from torch.nn.modules import padding
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class NiN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 192, kernel_size=5, stride=1, padding=2)
        self.conv1_2 = nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.drop1 = nn.Dropout2d(0.5)

        self.conv2_1 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
        self.conv2_2 = nn.Conv2d(192, 192, kernel_size=1,stride=1, padding=0)
        self.conv2_3 = nn.Conv2d(192, 192, kernel_size=1,stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(3, stride=2, padding=1)

        self.drop2 = nn.Dropout2d(0.5)

        self.conv3_1 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv3_3 = nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0)
        self.pool3 = nn.AvgPool2d(8, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)

        return x

        
batch_size = 32
transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])

fashion_mnist_trainval = FashionMNIST("FashionMNIST", train=True, download=True, transform=transform)
fashion_mnist_test = FashionMNIST("FashionMNIST", train=False, download=True, transform=transform)

n_samples = len(fashion_mnist_trainval) 
train_size = int(len(fashion_mnist_trainval) * 0.8) 
val_size = n_samples - train_size 

train_dataset, val_dataset = torch.utils.data.random_split(fashion_mnist_trainval, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers = 2)
test_loader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=True, num_workers = 2)

net = NiN()
net.to(device)

best_score = 100.0
count = 0
stop = 5

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
epoch_num = 200
running_loss = np.zeros(epoch_num)
val_running_loss = np.zeros(epoch_num)

for epoch in range(epoch_num):
  for i, data in enumerate(train_loader, 0):
    inputs, labels = data[0].to(device), data[1].to(device)
    optimizer.zero_grad() 
    outputs = net(inputs) 
    loss = criterion(outputs, labels) 
    loss.backward() 
    optimizer.step() 
    running_loss[epoch] += loss.item()
        
  for i, data in enumerate(val_loader, 0):    
    inputs, labels = data[0].to(device), data[1].to(device)
    optimizer.zero_grad() 
    outputs = net(inputs) 
    val_loss = criterion(outputs, labels) 
    val_loss.backward() 
    optimizer.step() 
    val_running_loss[epoch] += val_loss.item()
        
  running_loss[epoch] /= len(train_loader)
  val_running_loss[epoch] /= len(val_loader)

  if val_running_loss[epoch] > best_score:
    count += 1
    print("count:", count)

  else:
    count = 0
    best_score = val_running_loss[epoch]

  if count >= stop:
    print("early stopping")
    break

  print("epoch : %d, train_loss : %.4lf, val_loss : %.4lf" % (epoch, running_loss[epoch],val_running_loss[epoch]))
    
plt.figure()
plt.plot(running_loss[0:epoch+1])
plt.plot(val_running_loss[0:epoch+1])
plt.title('learning curve')
plt.savefig('%s.png' % ("learning curve"))


train_acc = 0.0
correct = 0.0
count = 0.0

with torch.no_grad():
  for i, data in enumerate(train_loader, 0):
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, pred_label = torch.max(outputs.data, 1)
    for j in range(len(pred_label)):
      if pred_label[j].int() == labels[j]:
        correct += 1
      count += 1

train_acc = correct/count
print("train_acc:",train_acc)



test_acc = 0.0
correct = 0.0
count = 0.0

with torch.no_grad():
  for i, data in enumerate(test_loader, 0):
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, pred_label = torch.max(outputs.data, 1)
    
  for j in range(len(pred_label)):
    if pred_label[j].int() == labels[j]:
      correct += 1  
    count += 1

test_acc = correct/count
print("test_acc:",test_acc)