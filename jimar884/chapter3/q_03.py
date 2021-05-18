import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

batch_size = 256
num_class = 10

def relu(x):
    x[x < 0] = 0
    return x

def softmax(x):
    for i in range(len(x)):
        x[i] = torch.exp(x[i]) / torch.sum(torch.exp(x[i]))
    return x

def cross_entropy(pred_y, y):
    tmp = torch.zeros((len(pred_y), num_class))
    for i in range(len(pred_y)):
        tmp[i, y[i].item()] = 1
    ans = torch.zeros(len(pred_y))
    for i in range(len(pred_y)):
        ans = torch.sum(-(tmp[i]*torch.log(pred_y[i])))
    return torch.mean(ans)

def whichclass(pred_y):
    ans = torch.zeros(len(pred_y))
    for i in range(len(pred_y)):
        ans[i] = torch.argmax(pred_y[i])
    return ans


def main():
    # donwload data & make DataLoader
    data_path = 'jimar884/chapter3/data'
    train_dataset = FashionMNIST(data_path, train=True, download=False, transform=transforms.ToTensor())
    test_dataset = FashionMNIST(data_path, train=False, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # initialize the params
    num, heignt, width = train_dataset.data.shape
    num_hiddenunit = 256

    w_in2hid = torch.normal(0, 0.01, size=(heignt*width, num_hiddenunit), requires_grad=True)
    b_in2hid = torch.zeros(num_hiddenunit, requires_grad=True)
    w_hid2out = torch.normal(0, 0.01, size=(num_hiddenunit, num_class), requires_grad=True)
    b_hid2out = torch.zeros(num_class, requires_grad=True)

    # train
    num_epoch = 10
    optimizer = optim.SGD([w_in2hid, b_in2hid, w_hid2out, b_hid2out], lr=0.01)
    losses = np.zeros(num_epoch)

    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, heignt*width)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                hiddenlayers = relu(torch.matmul(images, w_in2hid) + b_in2hid)
                outputs = torch.matmul(hiddenlayers, w_hid2out) + b_hid2out
                pred_labels = softmax(outputs)
                loss = cross_entropy(pred_labels, labels)
                losses[epoch] += loss / batch_size
            loss.backward()
            optimizer.step()
        print("epoch:%3d, loss:%.4f" % (epoch, losses[epoch]))

    # get accuracy
    correct = 0.0
    count = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.view(-1, heignt*width)
        hiddenlayers = torch.matmul(images, w_in2hid) + b_in2hid
        outputs = torch.matmul(hiddenlayers, w_hid2out) + b_hid2out
        pred_labels = softmax(outputs)
        pred_labels = whichclass(pred_labels)
        for j in range(len(pred_labels)):
            if pred_labels[j].int() == labels[j]:
                correct += 1
            count += 1
    acc = correct / count
    print("accuracy:%.4f" % (acc))
    
    # show loss
    plt.plot(losses)
    plt.show()

if __name__=='__main__':
    main()
