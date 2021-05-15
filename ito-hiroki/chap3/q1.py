import numpy as np
import math

from model import LinearRegression


def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum() / len(y_true)


BATCH_SIZE = 50
EPOCH_NUM = 100

X = np.load("X.npy")
y = np.load("noise_y.npy")

model = LinearRegression()

for epoch in range(EPOCH_NUM):
    indexes = np.random.permutation(len(X))
    # FIX: array_splitだと[1, 2, 3, 4, 5, 6, 7]を3分割すると
    # [1, 2, 3][4, 5][6, 7]になるので直感的ではない
    for batch_idx in np.array_split(indexes, math.ceil(len(X) / BATCH_SIZE)):
        Xs = X[batch_idx]
        ys = y[batch_idx]
        y_pred = model.update(Xs, ys)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"EPOCH {epoch}: weights: {model.weights}, bias: {model.bias}, mse: {mse}")