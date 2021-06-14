import torch.nn as nn
import torch
import numpy as np

a = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

print(a)

a.resize(6, 6)
print(a)