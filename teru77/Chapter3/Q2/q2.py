import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#Define  datasets
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST(root = './data',train=True,download=True,transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256,shuffle=True)

test_dataset = torchvision.datasets.FashionMNIST(root = './data',train=False,download=True,transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256,shuffle=False)

#Define a softmax operation
def softmax_operation(x):
    x_max,_ = torch.max(x,1,keepdim=True)
    e = torch.exp(x - x_max)
    u = torch.sum(e,1,keepdim=True)
    return e/u

#Define a model
class Net(nn.Module):
    def __init__(self,input_features,output_features):
        super().__init__() #nn.Moduleを継承
        self.linear = nn.Linear(input_features,output_features)
        nn.init.normal_(self.linear.weight, mean=0,std=0.01) #重みを初期化
        
    def forward(self,input):
        output = softmax_operation(self.linear(input))
        return output

model = Net(28*28,10)

#Define a cross-entropy loss
def cross_entropy_error(pred_y, y):
    return -1 * torch.mean(torch.sum( y * torch.log(pred_y),1))


#Define a stochastic gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)   
                         
epochs = 100
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
best_train_loss = None
best_train_acc = None
best_test_loss = None
best_test_acc = None
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
        images = images.reshape(-1,28*28) #Flattening
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        with torch.set_grad_enabled(True):
            pred_y = model(images)
            _, predicted = torch.max(pred_y.data, 1)
            loss=cross_entropy_error(pred_y,torch.nn.functional.one_hot(labels, num_classes=10)) #labels -> ワンホットベクトル化)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        #acc
        correct += torch.sum(predicted == labels).item()  
        total += labels.size(0) 
    print(f"loss:{total_loss/i}")
    train_loss = total_loss/i
    train_losses.append(train_loss) #epochごとの平均lossを格納
    train_acc = float(correct / total)
    train_accuracies.append(train_acc) #epochごとの平均accを格納
    
    if best_train_loss is None  or best_train_loss > train_loss:
        best_train_loss = train_loss
        best_train_acc =  train_acc
        
    #test
    total_loss =0
    correct = 0
    total = 0
    with torch.no_grad():
        for i,data in enumerate(test_dataloader,0):
            images, labels = data
            images = images.reshape(-1,28*28)
            pred_y = model(images)
            _, predicted = torch.max(pred_y.data, 1)
            #loss
            loss=cross_entropy_error(pred_y,torch.nn.functional.one_hot(labels, num_classes=10)) #labels -> ワンホットベクトル化)
            total_loss += loss.item()
            #acc
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_loss = total_loss/i
        test_losses.append(test_loss)
        test_acc = float(correct / total)
        test_accuracies.append(test_acc)
        
    if best_test_loss is None  or best_test_loss > test_loss:
        best_test_loss = test_loss
        best_test_acc =  test_acc

# Plot result(loss)
plt.plot(range(1, epochs+1),train_losses,label="train")
plt.plot(range(1, epochs+1),test_losses,label="test")
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('./q2_loss.png')
plt.close()

# Plot result(acc)
plt.plot(range(1, epochs+1),train_accuracies,label="train")
plt.plot(range(1, epochs+1),test_accuracies,label="test")
plt.title('Accuracies')
plt.xlabel('epoch')
plt.ylabel('Acc')
plt.legend()
plt.savefig('./q2_acc.png')
plt.close()
print("-"*30)
print(f"best_train_loss: {best_train_loss}")
print(f"best_train_acc: {best_train_acc}")
print(f"best_test_loss: {best_test_loss}")
print(f"best_test_acc: {best_test_acc}")
