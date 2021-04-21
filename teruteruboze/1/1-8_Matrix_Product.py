import numpy as np

if __name__ == '__main__':
    A = np.random.randint(1, 10, (5, 3))
    x = np.random.randint(1, 10, (3, 2))
    print(A)
    print(x)
    print(A @ x)   