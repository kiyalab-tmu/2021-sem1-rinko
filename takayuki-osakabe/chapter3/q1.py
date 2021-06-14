import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets, transforms

from utils.train import train
from utils.test import test
from utils.model import q1_Network

class Mydatasets(torch.utils.data.Dataset):
    def __init__(self):
        self.data = np.random.normal(0,1,(1000,2))
        self.label = np.dot(self.data, [2,-3.4]) + 4.2 + np.random.normal(0, 0.01, 1000)
        self.label = self.label.reshape((-1,1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

class SquaredLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, targets):
        loss = torch.mean((inputs - targets)**2)
        return loss

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(784))
    ])

train_loader = torch.utils.data.DataLoader(Mydatasets(), batch_size=32, shuffle=True)

model = q1_Network(2,1)
opt = optim.SGD(model.parameters(), lr=0.1)
criterion = SquaredLoss()

for epoch in range(100):
    train_acc, train_loss = train(model, train_loader, opt, criterion, epoch)

