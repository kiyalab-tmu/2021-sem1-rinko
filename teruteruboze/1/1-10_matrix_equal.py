import numpy as np

def isMatrixEqual(a,b):
    if len(a[a==b]) == len(a) and len(a) == len(b):
        return True
    return False

if __name__ == '__main__':
    A = np.random.randint(1, 3, 3)
    B = np.random.randint(1, 3, 3)
    print(A)
    print(B)

    # numpy function
    if np.allclose(A,B):
        print('True')

    # custom function
    if isMatrixEqual(A,B):
        print('True')