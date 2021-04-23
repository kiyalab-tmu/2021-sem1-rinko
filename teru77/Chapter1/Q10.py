import numpy as np

def Compare_Two_array():
    arrA = np.random.randint(0,2,2)
    print('First Array=',arrA)
    arrB = np.random.randint(0,2,2)
    print('Second Array=',arrB)
    print(np.all(arrA==arrB))
    
if __name__ == '__main__':
    Compare_Two_array()
    