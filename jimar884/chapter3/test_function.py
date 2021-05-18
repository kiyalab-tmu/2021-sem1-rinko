from numpy.core.numeric import zeros_like
import torch

a = torch.tensor(
    [[1, 2, 3, 4, 5],
    [2, 4, 6, 8, 10]]
)
b = torch.zeros(len(a))
for i in range(len(a)):
    b[i] = torch.argmax(a[i])
print(b)
