import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax

class Single_Layer_Network(nn.Module):
    def __init__(self, input_size=28*28, output_size=10):
        super(Single_Layer_Network, self).__init__()
        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(input_size, output_size)
        #self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
 
        # Not necessary when using nn.CrossEntropyLoss(). 
        # nn.CrossEntropyLoss() includes the softmax operation in their proccess.
        #x = self.softmax(x)
        return x

class Multilayer_Perceptrons(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=256, output_size=10):
        super(Multilayer_Perceptrons, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.Softmax(dim=-1)

    def weight_init(self, mu=0, sigma=1.0):
        nn.init.normal_(self.weight, mu, sigma)

    def weight_bias_init(self, mu=0, sigma=1.0):
        nn.init.normal_(self.weight, mu, sigma)
        nn.init.normal_(self.bias, mu, sigma)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
 
        # Not necessary when using nn.CrossEntropyLoss(). 
        # nn.CrossEntropyLoss() includes the softmax operation in their proccess.
        #x = self.softmax(x)
        return x

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=256, output_size=10):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2     = nn.Linear(hidden_size, hidden_size)
        self.fc3     = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.Softmax(dim=-1)

    def weight_init(self, mu=0, sigma=1.0):
        nn.init.normal_(self.weight, mu, sigma)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
 
        # Not necessary when using nn.CrossEntropyLoss(). 
        # nn.CrossEntropyLoss() includes the softmax operation in their proccess.
        #x = self.softmax(x)
        return x