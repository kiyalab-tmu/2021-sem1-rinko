import numpy as np

if __name__ == '__main__':
    A = np.random.randint(0, 15, 10)
    print('Before:', A)
    A = A[(A<3) | (8<A)]
    print('After :', A)