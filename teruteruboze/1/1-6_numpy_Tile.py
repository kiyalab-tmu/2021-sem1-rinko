import numpy as np

if __name__ == '__main__':
    n = 8
    row1 = np.tile(np.array([0, 1]), n//2)
    row2 = np.tile(np.array([1, 0]), n//2)
    out = np.array([])
    print(type(out))
    for i in range(n):
        if i % 2:
            out = np.append(out, row1)
        else:
            out = np.append(out, row2)       
    print(out.reshape([8,8]))