import torch
import numpy as np
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        torch.save(model, 'finish_model.pkl')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.output=nn.Linear(256,10)
        self.dropout=nn.Dropout(p=0.2)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x=self.hidden(x)
        #x=self.dropout(x)
        x = F.relu(x)
        x=self.output(x)
        x = F.softmax(x, dim=1)
        return x

batch_size=256
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=0,std=0.01)
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.FashionMNIST("dataset/", download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.FashionMNIST("dataset/", download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

image, label = next(iter(trainloader))

model = Classifier()
device = torch.device('cuda:0')
model.to(device)
#criterion = nn.CrossEntropyLoss()
#
criterion = nn.CrossEntropyLoss()

# 优化方法为Adam梯度下降方法，学习率为0.003
#optimizer = optim.SGD(model.parameters(), lr=0.001)

#optimizer = optim.Adam(model.parameters(), lr=0.003)
optimizer = optim.SGD(model.parameters(), lr=0.001,weight_decay=1e-2)
epoches = 150

print("train start")
es=EarlyStopping()
train_losses, test_losses = [], []
for i in range(epoches):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        images=images.to(device)
        labels=labels.to(device)
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    test_loss = 0
    accuracy = 0
    model.eval()
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        loss=criterion(out, labels)
        test_loss += loss

        values,index=torch.topk(input=out,k=1,dim=1)
        result=(index==labels.view(*index.shape))

        accuracy += torch.mean(result.type(torch.FloatTensor))



    train_losses.append(running_loss / len(trainloader))
    test_losses.append(test_loss / len(testloader))
    model.train()
    print("训练集学习次数: {}/{}.. ".format(i, epoches),
          "训练误差: {:.3f}.. ".format(running_loss / len(trainloader)),
          "测试误差: {:.3f}.. ".format(test_loss / len(testloader)),
          "模型分类准确率: {:.3f}".format(accuracy / len(testloader)))
    es(test_loss,model)
    if es.early_stop:
        print("Early stopping")
        # 结束模型训练
        break

# imagedemo=image[5].reshape((28,28))
# imagedemolabel=label[3]
# print(imagedemolabel)
# plt.imshow(imagedemo)
# plt.show()

#0.5 0.760
#0 0.768
#0.2 0.764

