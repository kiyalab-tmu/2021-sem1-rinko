import numpy as np

a = np.array(list(range(1, 11)))

print(a[(a < 3) | (a > 8)])
