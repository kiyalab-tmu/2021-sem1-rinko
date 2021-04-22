import numpy as np
import matplotlib.pyplot as plt


def checkerBoard(k: int = 8):
    line = [0 if x % 2 == 0 else 1 for x in range(k)]
    line2 = [1 if x % 2 == 0 else 0 for x in range(k)]
    board = np.tile([line, line2], (int(k / 2), 1))
    plt.imshow(np.expand_dims(board, axis=-1))
    plt.show()


if __name__ == "__main__":
    checkerBoard()
