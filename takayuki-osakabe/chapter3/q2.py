import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets, transforms

from utils.train import train
from utils.validation import validation
from utils.test import test
from utils.model import q2_Network
from utils.load_fashionmnist import load_fashionmnist

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(784))
    ])

train_loader, val_loader, test_loader = load_fashionmnist()

model = q2_Network(28*28,10)
opt = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

logs = []
for epoch in range(100):
    train_acc, train_loss = train(model, train_loader, opt, criterion, epoch)
    val_acc, val_loss = validation(model, val_loader, criterion)

    log = {'epoch':epoch, 'train_loss':train_loss, 'val_loss':val_loss, 'train_acc':train_acc, 'val_acc':val_acc}
    logs.append(log)

test_acc, _ = test(model, test_loader, criterion)
print(test_acc)

df = pd.DataFrame(logs)
df.to_csv('./log/q2_log.csv', index=False)
