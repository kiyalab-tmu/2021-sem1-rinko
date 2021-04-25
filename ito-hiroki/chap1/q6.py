import numpy as np

tmp = [0, 1]
ans = np.tile([tmp, tmp[::-1]], (4, 4))
print(ans)
