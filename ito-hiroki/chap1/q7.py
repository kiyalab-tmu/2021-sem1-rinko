import numpy as np

mat = np.random.rand(5, 5)
print(mat)
norm_mat = (mat - mat.mean()) / mat.std()
print(norm_mat)
