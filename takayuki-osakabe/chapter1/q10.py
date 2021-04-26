import numpy as np

a = np.random.randint(0,5,(3,2))
b = np.random.randint(0,5,(2,3))

print(np.all(a==a))
print(np.all(a==b))
