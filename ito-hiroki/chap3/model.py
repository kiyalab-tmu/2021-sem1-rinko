import numpy as np


class LinearRegression:
    def __init__(self, input_dim=2):
        np.random.seed(19980307)
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=2)
        self.bias = 0
        self.lr = 0.001

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def update(self, X, y):
        f = y - (np.dot(X, self.weights) + self.bias)
        f = f.reshape(-1, 1)
        self.weights -= self.lr * ((-2 * X * f).sum(axis=0) / len(y))
        self.bias -= self.lr * (-2 * f.sum() / len(y))
        return self.predict(X)
