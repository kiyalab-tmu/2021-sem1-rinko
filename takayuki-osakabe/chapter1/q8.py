import numpy as np

a = np.random.randint(0,10,(5,3))
print(a)
b = np.random.randint(0,10,(3,2))
print(b)
print(np.dot(a,b))
