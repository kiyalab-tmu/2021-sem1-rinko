import numpy as np
import random

a = np.arange(1,11)
np.random.shuffle(a)
a[(a > 3) & (a < 8)] *= -1
print(a)
