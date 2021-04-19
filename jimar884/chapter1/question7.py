import numpy as np

A = np.random.rand(5, 5)
print("before")
print(A)
max_A, min_A = A.max(), A.min()
A = (A - min_A) / (max_A - min_A)
print("after")
print(A)