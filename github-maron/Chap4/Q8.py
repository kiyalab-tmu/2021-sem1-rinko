#VGG
import torch
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

class VGG(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 64, kernel_size = 3, padding = 1)

    self.pool = nn.MaxPool2d(2, 2)

    self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)

    self.conv3= nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
    self.conv4= nn.Conv2d(256, 256, kernel_size = 3, padding = 1)

    self.conv5= nn.Conv2d(256, 512, kernel_size = 3, padding = 1)
    self.conv6= nn.Conv2d(512, 512, kernel_size = 3, padding = 1)

    self.conv7= nn.Conv2d(512, 512, kernel_size = 3, padding = 1)
    self.conv8= nn.Conv2d(512, 512, kernel_size = 3, padding = 1)

    self.fc1 = nn.Linear(7*7*512, 4096)
    self.fc2 = nn.Linear(4096, 4096)
    self.fc3 = nn.Linear(4096, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))

    x = self.pool(F.relu(self.conv2(x)))

    x = F.relu(self.conv3(x))
    x = self.pool(F.relu(self.conv4(x)))

    x = F.relu(self.conv5(x))
    x = self.pool(F.relu(self.conv6(x)))

    x = F.relu(self.conv7(x))
    x = self.pool(F.relu(self.conv8(x)))

    x = x.view(-1, 7*7*512)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
        
batch_size = 32
transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])

fashion_mnist_trainval = FashionMNIST("FashionMNIST", train=True, download=True, transform=transform)
fashion_mnist_test = FashionMNIST("FashionMNIST", train=False, download=True, transform=transform)

n_samples = len(fashion_mnist_trainval) 
train_size = int(len(fashion_mnist_trainval) * 0.8) 
val_size = n_samples - train_size 

train_dataset, val_dataset = torch.utils.data.random_split(fashion_mnist_trainval, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=True)

net = VGG()
net.to(device)

best_score = 100.0
count = 0
stop = 5

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002)
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