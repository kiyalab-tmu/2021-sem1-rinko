import numpy as np

def Multiply_matrix():
    arr1 = np.random.rand(5,3)
    print('First array=\n',arr1)
    
    arr2 = np.random.rand(3,2)
    print('Second array=\n',arr2)
    
    matrix_arr = np.dot(arr1,arr2)
    print('Matrix Product=\n',matrix_arr)
    
if __name__ == '__main__':
    Multiply_matrix()