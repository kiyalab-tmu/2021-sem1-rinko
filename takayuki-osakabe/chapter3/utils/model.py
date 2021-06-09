import torch
import torch.nn as nn
import numpy as np


def softmax(x):
    x_max, _ = torch.max(x,1,keepdim=True)
    a = torch.exp(x - x_max)
    b = torch.sum(a,1,keepdim=True)
    return a/b

def ReLU(x):
    x = x.detach().numpy()
    return torch.from_numpy(np.maximum(0,x))

class q1_Network(nn.Module):

    def __init__(self, input_size, output_size):
        super(q1_Network, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        nn.init.normal_(self.linear.weight, 0.0, 0.01)

    def forward(self, x):
        out = self.linear(x)
        return out

class q2_Network(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(q2_Network, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        nn.init.normal_(self.linear.weight, 0.0, 0.01)

    def forward(self, x):
        x = self.linear(x)
        return softmax(x)

class q3_Network(nn.Module):

    def __init__(self, input_size, output_size):
        super(q3_Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256,output_size)
        self.relu = nn.ReLU()
        nn.init.normal_(self.fc1.weight, 0.0, 0.01)
        nn.init.normal_(self.fc2.weight, 0.0, 0.01)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return softmax(x)

class q4_dropout(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(q4_dropout, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        nn.init.normal_(self.fc1.weight, 0.0, 0.01)
        nn.init.normal_(self.fc2.weight, 0.0, 0.01)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return softmax(x)
