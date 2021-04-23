import numpy as np 

def Normalize_random_matrix():
    arr = np.random.rand(5,5)  
    max,min = arr.max(),arr.min()
    Normalize_arr = (arr-min)/(max-min)
    
    return arr,Normalize_arr

if __name__ == '__main__':
    Original,Normalize = Normalize_random_matrix()
    print('Original=\n{}\nNormalize=\n{}'.format(Original,Normalize))