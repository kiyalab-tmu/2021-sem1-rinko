import numpy as np

def Negate_elements():
    arr = np.random.randint(0,10,10)
    arr[(3 < arr) & (arr < 8)] *= -1
    print(arr)
    
if __name__ == '__main__':
    Negate_elements()