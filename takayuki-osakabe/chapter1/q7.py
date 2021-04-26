import numpy as np

array = np.random.randint(0,25,(5,5))
res = (array - array.mean()) / array.std()
print(array)
print(res)
