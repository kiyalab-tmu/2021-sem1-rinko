import torch
import torchvision
import torch.nn as nn

class NNet(nn.Module):
    def __init__(self, num_attributes):
        super(NNet, self).__init__()
        self.classifier = torchvision.models.resnet18(pretrained=False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_attributes)
    def forward(self, xb):
        y = self.classifier(xb).sigmoid()
        return y

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            gen_block(z_dim,1024,4,1,0),
            gen_block(1024,512,4,2,1),
            gen_block(512,256,4,2,1),
            gen_block(256,128,4,2,1),
            gen_block(128,3,4,2,1,final_layer=True))
        weights_init(self.layers)

    def forward(self, noise):
        return self.layers(noise)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def gen_block(input_channels, output_channels, kernel_size, stride, padding, final_layer=False):
    if not final_layer:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride,padding),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(True))
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride,padding),
            nn.Tanh())