import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets, transforms

from utils.train import train
from utils.test import test
from utils.model import q3_Network

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(784))
    ])
train_dataset = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

test_dataset = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256,shuffle=False)

model = q3_Network(28*28,10)
opt = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

logs = []
for epoch in range(100):
    train_acc, train_loss = train(model, train_loader, opt, criterion, epoch)
    test_acc, test_loss = test(model, test_loader, criterion)

    log = {'epoch':epoch, 'train_loss':train_loss, 'test_loss':test_loss, 'train_acc':train_acc, 'test_acc':test_acc}
    logs.append(log)

df = pd.DataFrame(logs)
df.to_csv('./log/q4_stopping_log.csv', index=False)
