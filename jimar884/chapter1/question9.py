import numpy as np

A = np.arange(10)
A[(3 < A) & (A < 8)] *= -1
print(A)