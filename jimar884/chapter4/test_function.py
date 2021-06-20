import torch.nn as nn
import torch
import numpy as np

m = nn.AdaptiveAvgPool2d((5, 7))
input = torch.randn(1, 64, 8, 9)
output = m(input)
print(input.shape)
print(output.shape)