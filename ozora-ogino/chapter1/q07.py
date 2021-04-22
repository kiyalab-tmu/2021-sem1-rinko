import numpy as np


def minMaxNorm(xs: np.ndarray):
    mi, mx = min(xs.flatten()), max(xs.flatten())
    return (xs - mi) / (mx - mi)


if __name__ == "__main__":
    matrix = np.random.rand(10, 10)
    print(minMaxNorm(matrix))
