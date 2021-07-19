import random
import torch

# The true parameter
W = torch.tensor([2, -3.4])
B = 4.2

# Generate a synthetic dataset
N = 1000     # number of features
features = torch.normal(0, 1, (N, len(W)))     # 1000x2
labels = torch.matmul(features, W) + B
labels += torch.normal(0, 0.01, labels.shape)
# labels = labels.reshape((-1,1))     # 1000x1

# batch
def data_iter(features, labels, batch_size=4):
    num = len(features)
    index = list(range(num))
    random.shuffle(index)
    for i in range(0, num, batch_size):
        batch_index = torch.tensor(index[i:min(i+batch_size, num)])
        yield features[batch_index], labels[batch_index]

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y, pred_y):
    return (pred_y - y.reshape(pred_y.shape))**2 / 2

def sgd(params, lr, batch_size=4):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter(features, labels):
        loss = squared_loss(y, linreg(X, w, b))
        loss.sum().backward()
        sgd([w,b], lr)
    with torch.no_grad():
        train_loss = squared_loss(labels, linreg(features, w, b))
        print(f'epoch {epoch + 1}, loss {float(train_loss.mean()):f}')

# print(f'error in estimating w: {W - w.reshape(W.shape)}')
# print(f'error in estimating b: {B - b}')

print(w, b)