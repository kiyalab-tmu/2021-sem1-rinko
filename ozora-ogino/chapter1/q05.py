import numpy as np


def multiplicationTable(k: int) -> np.ndarray:
    table = np.ones([k, k])
    for i in range(k):
        table[i] *= i + 1
        table[:, i] *= i + 1
    return table.astype(int)


def printTable(mat: np.ndarray, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


if __name__ == "__main__":
    k = 10
    table = multiplicationTable(k)
    printTable(table)
