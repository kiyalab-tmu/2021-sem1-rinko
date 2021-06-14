import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets, transforms

from utils.load_fashionmnist import load_fashionmnist
from utils.train import train
from utils.validation import validation
from utils.test import test
from utils.model import LeNet


train_loader, val_loader, test_loader = load_fashionmnist(batch_size=64)

model = LeNet()
opt = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

logs = []
for epoch in range(50):
    train_acc, train_loss = train(model, train_loader, opt, criterion, epoch)
    val_acc, val_loss = validation(model, val_loader, criterion)

    log = {'epoch':epoch, 'train_loss':train_loss, 'val_loss':val_loss, 'train_acc':train_acc, 'val_acc':val_acc}
    logs.append(log)

torch.save(model.state_dict(), './trained_model/q6.pth')

test_acc, _ = test(model, test_loader, criterion)
print(test_acc)

df = pd.DataFrame(logs)
df.to_csv('./log/q6_log.csv', index=False)
