import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = "cuda:0" if (torch.cuda.is_available()) else "cpu"
print(device)
#Define  datasets
transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST(root = './data',train=True,download=True,transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256,shuffle=True)

test_dataset = torchvision.datasets.FashionMNIST(root = './data',train=False,download=True,transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256,shuffle=False)

#Define a model
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(5*5*256, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = AlexNet()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 30
train_losses = []
train_accuracies = []
best_train_loss = None
best_train_acc = None

for epoch in range(epochs):  # loop over the dataset multiple times
    print("-"*30)
    print("epoch : {}".format(epoch))
    
    #train
    model.train()
    total_loss =0
    correct = 0
    total = 0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [X, y]
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        with torch.set_grad_enabled(True):
            pred_y = model(images)
            _, predicted = torch.max(pred_y, 1)
            loss=criterion(pred_y,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        #acc
        correct += torch.sum(predicted == labels.data).item()  
        total += labels.size(0) 
    print(f"loss: {total_loss/i}")
    train_loss = total_loss/i
    train_losses.append(train_loss) #epochごとの平均lossを格納
    train_acc = float(correct / total)
    print(f"acc: {train_acc}")
    train_accuracies.append(train_acc) #epochごとの平均accを格納
    
    if best_train_loss is None  or best_train_loss > train_loss:
        best_train_loss = train_loss
        best_train_acc =  train_acc

torch.save(model.state_dict(), './model.pth')
  
#test
model = AlexNet()
model.load_state_dict(torch.load('./model.pth'))    
model = model.to(device)
model.eval()  

total_loss =0
correct = 0
total = 0
with torch.no_grad():
    for i,data in enumerate(test_dataloader,0):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        pred_y = model(images)
        _, predicted = torch.max(pred_y, 1)
        #loss
        loss=criterion(pred_y,labels)
        total_loss += loss.item()
        #acc
        total += labels.size(0)
        correct += (predicted == labels.data).sum().item()
    test_loss = total_loss/i
    test_acc = float(correct / total)
       
        
# Plot result(loss)
plt.plot(range(1, epochs+1),train_losses,label="train")
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('./q7_loss.png')
plt.close()

# Plot result(acc)
plt.plot(range(1, epochs+1),train_accuracies,label="train")
plt.title('Accuracies')
plt.xlabel('epoch')
plt.ylabel('Acc')
plt.legend()
plt.savefig('./q7_acc.png')
plt.close()
print("-"*30)
print(f"best_train_loss: {best_train_loss}")
print(f"best_train_acc: {best_train_acc}")
print(f"test_loss: {test_loss}")
print(f"test_acc: {test_acc}")
