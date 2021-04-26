import numpy as np

a = np.array(list(range(1, 11)))

a[(a >= 3) & (a <= 8)] *= -1
print(a)
