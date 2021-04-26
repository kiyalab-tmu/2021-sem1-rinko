import numpy as np

def Create_Checkerboard():
    return np.tile([[0,1],[1,0]],(4,4))

if __name__ == '__main__':
    print(Create_Checkerboard())