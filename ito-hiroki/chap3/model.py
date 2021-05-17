import numpy as np


class LinearRegression:
    def __init__(self, input_dim=2):
        np.random.seed(19980307)
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=input_dim)
        self.bias = 0
        self.lr = 0.01

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def update(self, X, y):
        f = y - (np.dot(X, self.weights) + self.bias)
        f = f.reshape(-1, 1)
        self.weights -= self.lr * ((-2 * X * f).sum(axis=0) / len(y))
        self.bias -= self.lr * (-2 * f.sum() / len(y))
        return self.predict(X)


class SoftmaxRegression:
    def __init__(self, input_dim=784, output_dim=10):
        np.random.seed(19980307)
        self.weights = np.random.normal(
            loc=0.0, scale=0.01, size=(input_dim, output_dim)
        )
        self.bias = np.zeros((1, 10))
        self.lr = 0.01

    def predict(self, X):
        X = np.dot(X, self.weights) + self.bias
        return self.softmax(X)

    def update(self, X, y):
        pred = self.predict(X)
        # f = -np.mean(y * np.log(pred.T + 1e-8))
        dZ = pred - y
        dW = np.dot(X.T, dZ) / len(X)
        db = np.sum(dZ, axis=0, keepdims=True) / len(X)

        self.weights -= self.lr * dW
        self.bias -= self.lr * db

    @staticmethod
    def softmax(x):
        exp = np.exp(x - np.max(x))
        return exp / exp.sum(axis=1, keepdims=True)
        # return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
