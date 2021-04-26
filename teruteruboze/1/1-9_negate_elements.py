import numpy as np

if __name__ == '__main__':
    A = np.random.randint(0, 15, 10)
    print('Before:', A)
    A[(3<=A) & (A<=8)] = A[(3<=A) & (A<=8)] * -1
    print('After :', A)