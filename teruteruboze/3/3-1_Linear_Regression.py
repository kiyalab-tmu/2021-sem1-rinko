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

    def f(self, X):
        return (X @ self.W) + self.b

    # X: data X, Y: data Y (train data)
    def f_loss(self, Y, X):
        return self.squared_loss(Y, self.f(X))

    def squared_loss(self, y, y_pred):
        return np.mean((y - y_pred)**2)

    # grad群
    def grad(self, Y, X, h = 1e-4):
        if   self.W.ndim == 1:
            return self.grad_1d(Y, X, h)
        else:
            raise ValueError('Not support other than dim = 1')

    def grad_1d(self, y, x, h):
        grad = np.zeros_like(self.W)
        for idx in range(self.W.size):
            tmp_val = self.W[idx]
            self.W[idx] = float(tmp_val) + h
            fxh1 = self.f_loss(y, x) # f(x+h)
            
            self.W[idx] = float(tmp_val) - h 
            fxh2 = self.f_loss(y, x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h) # +方向の微分と-方向の微分の差
            
            self.W[idx] = tmp_val # 値を元に戻す

        if self.b != 0:
            tmp_val = self.b
            self.b = float(tmp_val) + h
            fxh1 = self.f_loss(y, x) # f(x+h)
            
            self.b = float(tmp_val) - h 
            fxh2 = self.f_loss(y, x) # f(x-h)
            grad_bias = (fxh1 - fxh2) / (2*h) # +方向の微分と-方向の微分の差
            
            self.b = tmp_val # 値を元に戻す
            
            return grad, grad_bias

        else:
            return grad

def GradientDescent(W, dL, lr = 1e-3):
    return W - (lr * dL)

def plot(X, y, net, f_name):
    X_lin = np.array([np.linspace(np.min(X[:,0]), np.max(X[:,0]), 1000),
                      np.linspace(np.min(X[:,1]), np.max(X[:,1]), 1000)]).T
    fig = plt.figure()

    X_lin_f_in = zeros_like(X_lin)
    X_lin_f_in[:,0] = X_lin[:,0]
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(X[:,0], y)
    ax1.plot(X_lin[:,0], net.f(X_lin_f_in), color=(0.8,0.2,0.1))
    ax1.set_xlabel('x1')
    ax1.set_ylabel('y')
    
    X_lin_f_in = zeros_like(X_lin)
    X_lin_f_in[:,1] = X_lin[:,1]
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(X[:,1], y)
    ax2.plot(X_lin[:,1], net.f(X_lin_f_in), color=(0.8,0.2,0.1))
    ax2.set_xlabel('x2')

    fig.savefig(f_name)

def plot_loss(data, xlabel, ylabel, path):
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(list(range(len(data))), data, label='loss')
    ax1.set_xlabel(xlabel)
    ax1.legend()
    ax1.set_ylabel(ylabel)
    ax1.set_title(ylabel + ' / epoch')
    fig.savefig(path)

if __name__ == '__main__':
    BATCH_SIZE = 100
    data = BasicDataset(np.array([2, -3.4]), 4.2)
    net = Linear_Regression_Model(data.W.shape, 1)

    train_loss = []
    for epoch in range(500):
        for i in range(0, len(data.X), BATCH_SIZE):
            x_batch = data.X[i:i+BATCH_SIZE, :]
            y_batch = data.y[i:i+BATCH_SIZE]
            loss = net.f_loss(y_batch, x_batch)
            grad, grad_b = net.grad(y_batch, x_batch)

            net.W = GradientDescent(net.W, grad)
            net.b = GradientDescent(net.b, grad_b)
        
        train_loss.append(loss)       
        if epoch % 10 == 0:
            print(net.W)
            print(net.b)
            #plot(data.X, data.y_true, net, '3-1_Linear_Regression/per epoch/'+str(epoch+1)+'.jpg')

    print('============== analysis ===================')
    print('answer W:', np.array([2, -3.4]))
    print('pred   W:', net.W, '\n')
    print('answer sum(W):', np.sum([2, -3.4]))
    print('pred   sum(W):', np.sum(net.W))
    print('answer bias:', 4.2)
    print('pred   bias:', net.b)
    print('loss(y_true, y_pred):', net.f_loss(data.y_true, data.X))

    plot(data.X, data.y, net, '3-1_Linear_Regression/y.jpg')
    plot(data.X, data.y_true, net, '3-1_Linear_Regression/y_true.jpg')
    plot_loss(train_loss, 'epoch', 'MSE loss', '3-1_Linear_Regression/loss.jpg')