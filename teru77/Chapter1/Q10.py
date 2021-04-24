import numpy as np

def Compare():
    arrA = np.random.randint(0,2,2)
    print('First Array=',arrA)
    arrB = np.random.randint(0,2,2)
    print('Second Array=',arrB)
    print(np.all(arrA==arrB))
    
if __name__ == '__main__':
    Compare()
    
