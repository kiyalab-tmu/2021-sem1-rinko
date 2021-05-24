import torch
import numpy as np
from matplotlib import pyplot as plt


def squaredloss(batch_y, z, n):
    loss = 0

    for i in range(len(batch_y)):
        loss += (batch_y[i] - z[i]) ** 2
    loss /= len(batch_y)
    return loss


def minibatch(x, y, batch_size, train_num):
    permutation = list(np.random.permutation(train_num))
    shuffled_x = train_x[permutation, :]
    shuffled_y = train_y[permutation, :]

    batches_x = []
    batches_y = []
    for i in range(0, train_num, batch_size):
        if (i + batch_size >= train_num):
            end = train_num
        else:
            end = i + batch_size
        batch_x = shuffled_x[i:(i + batch_size),:]
        batch_y = shuffled_y[i:(i + batch_size),:]
        #batch_x = train_x[i:(i + batch_size), :]
        #batch_y = train_y[i:(i + batch_size), :]
        batches_x.append(batch_x)
        batches_y.append(batch_y)
    return batches_x, batches_y


def gradient(batch_x, batch_y, z):
    temp = z - batch_y
    temp2 = torch.mul(temp, batch_x)
    dw2 = 2*torch.mean(temp2, dim=0)
    db2 = 2*torch.mean(temp)
    return dw2, db2


true_w = torch.tensor([2, -3.4])
train_num = 1000
true_b = 4.2
train_x = torch.randn(train_num, 2)
x = true_w.resize(2, 1)
true_data = torch.mm(train_x, true_w.resize(2, 1)) + true_b
train_y = true_data + torch.normal(0, 0.01, true_data.size())

plt.scatter(train_x[:, 0], train_y)
plt.scatter(train_x[:, 1], train_y)
plt.plot([-4, 4], [-4 * 2 + 4.2, 4 * 2 + 4.2], color='red')
plt.plot([-4, 4], [-4 * -3.4 + 4.2, 4 * -3.4 + 4.2], color='red')
plt.show()

pre_w = torch.rand((1, 2))
pre_b = torch.rand((1, 1))

batch_size = 64

alpha = 0.001
epoches = 200
lossrecord = np.zeros(epoches)
for i in range(epoches):
    batches_x, batches_y = minibatch(train_x, train_y, batch_size, train_num)
    for j in range(len(batches_x)):
        z = torch.mm(batches_x[j], pre_w.resize(2, 1)) + pre_b
        loss = squaredloss(batches_y[j], z, 2)
        dw, db = gradient(batches_x[j], batches_y[j], z)
        pre_w -= dw * alpha
        pre_b -= db * alpha
    lossrecord[i] = loss
    print(pre_w, pre_b)
plt.plot(np.arange(epoches), lossrecord)
plt.xlabel("epoches")
plt.ylabel("loss")
plt.show()
