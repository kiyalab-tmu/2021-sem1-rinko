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

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1    = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.conv2    = nn.Conv2d(6, 16, 5)
        self.sigmoid  = nn.Sigmoid()
        self.avePool1 = nn.AvgPool2d(2, stride=2)
        self.avePool2 = nn.AvgPool2d(2, stride=2)
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(5 * 5 * 16, 120)
        self.fc2      = nn.Linear(120, 84)
        self.fc3      = nn.Linear(84, 10)
    
    def weight_init(self, mu=0, sigma=1.0):
        nn.init.normal_(self.weight, mu, sigma)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.avePool1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.avePool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class LeNet_BN(nn.Module):
    def __init__(self):
        super(LeNet_BN, self).__init__()
        self.conv1    = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.conv2    = nn.Conv2d(6, 16, 5)
        self.bn1      = nn.BatchNorm2d(6, eps=1e-05, momentum=0.9)
        self.bn2      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.9)
        self.sigmoid  = nn.Sigmoid()
        self.avePool1 = nn.AvgPool2d(2, stride=2)
        self.avePool2 = nn.AvgPool2d(2, stride=2)
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(5 * 5 * 16, 120)
        self.fc2      = nn.Linear(120, 84)
        self.fc3      = nn.Linear(84, 10)
    
    def weight_init(self, mu=0, sigma=1.0):
        nn.init.normal_(self.weight, mu, sigma)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sigmoid(x)
        x = self.avePool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        x = self.avePool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1    = nn.Conv2d(1, 96, 11, stride=4)
        self.relu     = nn.ReLU(inplace=True)
        self.maxPool1 = nn.MaxPool2d(3, stride=2)
        self.conv2    = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.maxPool2 = nn.MaxPool2d(3, stride=2)
        self.conv3    = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv4    = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv5    = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.maxPool3 = nn.MaxPool2d(3, stride=2)
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(5 * 5 * 256, 4096)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2      = nn.Linear(4096, 4096)
        self.fc3      = nn.Linear(4096, 10)
    
    def weight_init(self, mu=0, sigma=1.0):
        nn.init.normal_(self.weight, mu, sigma)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxPool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxPool3(x)
        print(x.shape)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.conv1    = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2    = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3    = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4    = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5    = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv6    = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv7    = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv8    = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.maxPool1 = nn.MaxPool2d(2, stride=2)
        self.maxPool2 = nn.MaxPool2d(2, stride=2)
        self.maxPool3 = nn.MaxPool2d(2, stride=2)
        self.maxPool4 = nn.MaxPool2d(2, stride=2)
        self.maxPool5 = nn.MaxPool2d(2, stride=2)
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(7 * 7 * 512, 4096)
        self.fc2      = nn.Linear(4096, 1000)
        self.fc3      = nn.Linear(1000, 10)
        self.relu     = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
    
    def weight_init(self, mu=0, sigma=1.0):
        nn.init.normal_(self.weight, mu, sigma)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxPool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxPool3(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.maxPool4(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.relu(x)
        x = self.maxPool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class NiN_Blocks(nn.Module):
    def __init__(self):
        super(NiN_Blocks, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 192, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(inplace=True),

            nn.Conv2d(96, 192, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(inplace=True),

            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 10, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(10)
        )
    
    def weight_init(self, mu=0, sigma=1.0):
        nn.init.normal_(self.weight, mu, sigma)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.avg_pool2d(x, 7, stride=1, padding=0)
        x = x.view(x.size(0), 10)
        return x

class _NiN_Blocks(nn.Module):
    def __init__(self):
        super(NiN_Blocks, self).__init__()
        self.features = nn.Sequential(
            self.nin_block(1,96,  kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(96,256,  kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(256,384,  kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(inplace=True),
            self.nin_block(384, 10, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    # コピペはいいけど，selfは忘れない！
    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        blk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(out_channels, out_channels, 1, 1, 1),
            nn.ReLU(inplace=True),
        
            nn.Conv2d(out_channels, out_channels, 1, 1, 1),
            nn.ReLU(inplace=True)
            )
        return blk
    
    def weight_init(self, mu=0, sigma=1.0):
        nn.init.normal_(self.weight, mu, sigma)

    def forward(self, x):
        x = self.features(x)
        return x