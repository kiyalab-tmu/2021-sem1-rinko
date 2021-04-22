import numpy as np
from pprint import pprint


def dot(xs1: np.ndarray, xs2: np.ndarray):
    if xs1.shape[1] != xs2.shape[0]:
        raise Exception("Inputs shape is not match")
    matrix = np.ones((xs1.shape[0], xs2.shape[1]))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i][j] = sum([v1 * v2 for v1, v2 in zip(xs1[i], xs2.T[j])])
    return matrix


if __name__ == "__main__":
    x = np.random.random((5, 3))
    print("First array:")
    pprint(x)
    y = np.random.random((3, 2))
    print("Second array:")
    pprint(y)
    print("Dot:")
    pprint(dot(x, y))
