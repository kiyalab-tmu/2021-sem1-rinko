import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import zeros_like

class BasicDataset:
    def __init__(self, W, b, num_data=1000):
        self.W = W
        self.b = b
        self.X = self.makeXdata(num_data)
        self.y_true = (self.X @ self.W.T) + self.b
        self.y = self.addNoise(self.y_true, num_data)

    def makeXdata(self, num_data, start=-5, end=5,):
        return np.random.randn(1000, 2)

    def addNoise(self, Data, num_data, mu=0, sigma=1):
        return Data + np.random.normal(mu, sigma, num_data)

class Linear_Regression_Model:
    def __init__(self, shape_W, shape_b=0, mu=0, sigma=1):
        self.W = np.random.normal(mu, sigma, shape_W)
        if shape_b != 0:
            self.b = np.random.normal(mu, sigma)
        else:
            self.b = 0