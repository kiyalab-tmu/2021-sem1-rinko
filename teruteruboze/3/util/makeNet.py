import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax

class Single_Layer_Network(nn.Module):
    def __init__(self, input_size=28*28, output_size=10):
        super(Single_Layer_Network, self).__init__()
        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
 
        # Not necessary when using nn.CrossEntropyLoss(). 
        # nn.CrossEntropyLoss() includes the softmax operation in their proccess.
        #x = self.softmax(x)
        return x