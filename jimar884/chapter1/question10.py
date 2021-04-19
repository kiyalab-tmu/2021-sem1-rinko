import numpy as np

A = np.random.rand(3,3)
B = np.random.rand(3,3)
print(np.all(A==A))
print(np.all(A==B))
