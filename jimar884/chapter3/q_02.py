from keras.datasets import fashion_mnist
import numpy as np
import random
import torch

# load data
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
# check shape and type
print(train_x.shape, test_x.shape)
print(len(train_y))
print(type(train_x))

# reshape
train_x  = train_x / 255.0
test_x = test_x / 255.0
train_x = train_x.reshape(60000, 28*28)
test_x = test_x.reshape(10000, 28*28)
# check shape and type
print(train_x.shape, test_x.shape)
print(type(train_x))

# One Hot Encoding
def one_hot(y, c=10):
    y_hot = np.zeros((len(y), c))
    for i in range(len(y)):
        y_hot[i, y[i]] = 1
    return y_hot

# softmax function
def softmax(z):
    exp = np.exp(z - np.max(z))
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
    return exp

# cross entropy
def cross_entropy(y, y_hat):
    return -np.mean(np.log(y_hat[np.arange(len(y)), y]))

# batch
def data_iter(x, y, batch_size=256):
    num = len(x)
    index = list(range(num))
    random.shuffle(index)
    for i in range(0, num, batch_size):
        batch_index = torch.tensor(index[i:min(i+batch_size, num)])
        yield x[batch_index], y[batch_index]

# train
def fit(x, y, c=10, lr=0.1, epochs=100):
    m, n = x.shape
    w = np.random.normal(loc=0, scale=0.01, size=(n,c))
    b = np.zeros_like(c)

    losses = []

    for epoch in range(epochs):
        for x_batch, y_batch in data_iter(x, y):
            z = x_batch@w + b
            y_hat = softmax(z)
            y_hot = one_hot(y_batch, c)
            w_grad = (1/m)*np.dot(x_batch.T, (y_hat - y_hot))
            b_grad = (1/m)*np.sum(y_hat - y_hot)
            w = w - lr*w_grad
            b = b - lr*b_grad
            loss = cross_entropy(y_batch, y_hat)
            losses.append(loss)
        if epoch%10==0:
            print('epoch {epoch} -> loss = {loss}'.format(epoch=epoch,loss=loss))
    
    return w, b, losses

w, b, l = fit(train_x, train_y)

# prediction
def predict(x, w, b):
    z = x@w + b
    y_hat = softmax(z)
    return np.argmax(y_hat,axis=1)

def accuracy(y, y_hat):
    return np.sum(y==y_hat)/len(y)

train_pred = predict(train_x, w, b)
print(accuracy(train_y, train_pred))
test_pred = predict(test_x, w, b)
print(accuracy(test_y, test_pred))

