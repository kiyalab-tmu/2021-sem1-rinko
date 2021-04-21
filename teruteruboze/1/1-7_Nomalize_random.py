import numpy as np

if __name__ == '__main__':
    matrix = np.random.rand(5, 5)
    print(matrix / matrix.max())   