import torch
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 5, padding = 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
batch_size = 32
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5),(0.5))])

fashion_mnist_trainval = FashionMNIST("FashionMNIST", train=True, download=True, transform=transform)
fashion_mnist_test = FashionMNIST("FashionMNIST", train=False, download=True, transform=transform)

n_samples = len(fashion_mnist_trainval) 
train_size = int(len(fashion_mnist_trainval) * 0.8) 
val_size = n_samples - train_size 

train_dataset, val_dataset = torch.utils.data.random_split(fashion_mnist_trainval, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=True)

net = LeNet()
net.to(device)

best_score = 100.0
count = 0
stop = 5

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002)
epoch_num = 200
running_loss = np.zeros(epoch_num)
val_running_loss = np.zeros(epoch_num)
train_acc = np.zeros(epoch_num)
val_acc = np.zeros(epoch_num)

for epoch in range(epoch_num):

  train_correct = 0
  train_count = 0
  val_correct = 0
  val_count = 0

  for i, data in enumerate(train_loader, 0):
    inputs, labels = data[0].to(device), data[1].to(device)
    optimizer.zero_grad() 
    outputs = net(inputs) 
    loss = criterion(outputs, labels) 
    loss.backward() 
    optimizer.step() 
    running_loss[epoch] += loss.item()

    _, pred_label = torch.max(outputs.data, 1)
    for j in range(len(pred_label)):
      if pred_label[j].int() == labels[j]:
        train_correct += 1
      train_count += 1

   
  for i, data in enumerate(val_loader, 0):    
    inputs, labels = data[0].to(device), data[1].to(device)
    optimizer.zero_grad() 
    outputs = net(inputs) 
    val_loss = criterion(outputs, labels) 
    val_loss.backward() 
    optimizer.step() 
    val_running_loss[epoch] += val_loss.item()
    
    _, pred_label = torch.max(outputs.data, 1)
    for j in range(len(pred_label)):
      if pred_label[j].int() == labels[j]:
        val_correct += 1
      val_count += 1

  train_acc[epoch] = train_correct/train_count
  val_acc[epoch] = val_correct/val_count
        
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

  print("epoch : %d, train_loss : %.4lf, val_loss : %.4lf, train_acc : %.4lf, val_acc : %.4lf" 
  % (epoch, running_loss[epoch],val_running_loss[epoch], train_acc[epoch], val_acc[epoch]))
    
plt.figure()
plt.plot(running_loss[0:epoch+1], label = "train_loss")
plt.plot(val_running_loss[0:epoch+1], label = "val_loss")
plt.legend()
plt.title('learning curve')
plt.savefig('%s.png' % ("Q11_loss"))

plt.figure()
plt.plot(train_acc[0:epoch+1], label = "train_acc")
plt.plot(val_acc[0:epoch+1], label = "val_acc")
plt.legend()
plt.title('learning curve')
plt.savefig('%s.png' % ("Q11_acc"))


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