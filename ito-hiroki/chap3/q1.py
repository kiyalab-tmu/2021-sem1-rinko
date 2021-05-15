import numpy as np

from model import LinearRegression


def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).sum() / len(y_true)


BATCH_SIZE = 50
EPOCH_NUM = 200

X = np.load("X.npy")
y = np.load("noise_y.npy")

model = LinearRegression()

for epoch in range(EPOCH_NUM):
    indexes = np.random.permutation(len(X))
    for batch_idx in np.array_split(indexes, BATCH_SIZE):
        Xs = X[batch_idx]
        ys = y[batch_idx]
        y_pred = model.update(Xs, ys)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"EPOCH {epoch}: weights: {model.weights}, bias: {model.bias}, mse: {mse}")