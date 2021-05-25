import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import datasets, transforms


class MultilayerPerceptrons(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultilayerPerceptrons, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.ReLU = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        nn.init.normal_(self.linear1.weight, 0.0, 0.01)
        nn.init.normal_(self.linear2.weight, 0.0, 0.01)

    def forward(self, x):
        x = self.linear1(x)
        x = self.ReLU(x)
        #x = self.dropout(x)
        x = self.linear2(x)
        out = F.softmax(x, dim=-1)
        return out


class early_stopping():
    def item(loss_now, loss_before, count):
        if loss_now > loss_before:
            count += 1
        else:
            count = 0
        return loss_now, count

    def early_stop(count):
        if count==3:
            return True
        else:
            return False


def mnist_learning(hidden_size=256, lr=0.01, batch_size=256, n_epoch=10, debug=True, fashion=True):
    # ----- MODEL SETTING -----
    model = MultilayerPerceptrons(input_size=784, hidden_size=256, output_size=10)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(784))
    ])

    # ----- PREPROCESS -----
    # ----- MNIST -----
    if not fashion: 
        dataloader_train = torch.utils.data.DataLoader(
            datasets.MNIST('./data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=True
        )
        dataloader_test = torch.utils.data.DataLoader(
            datasets.MNIST('./data/mnist', train=False, download=True, transform=transform),
            shuffle=False
        )
    # ----- FASHION-MNIST -----
    elif fashion:
        dataloader_train = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=True
        )
        dataloader_test = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True, transform=transform),
            shuffle=False
        )

    # ----- SHOW SAMPLE IMAGE -----
    if debug:
        data_train_0 = dataloader_train.dataset[0]
        x_train_0 = data_train_0[0].numpy().reshape(28, 28)
        t_train_0 = data_train_0[1]
        print("This Is Debug Mode. Show Sample Image...\n")
        print("DATA No.0 is {}".format(t_train_0))
        plt.imshow(x_train_0, cmap=cm.Greys)
        plt.show();

    # ----- LEARNING -----
    loss_train_list = []
    loss_test_list = []
    acc_test_list = []

    loss_before = 100
    stop_count = 0
    for epoch in range(1, n_epoch+1):
        loss_train_batch_list = []
        loss_test_batch_list = []
        correct = 0
        total = 0

        if epoch == 30:
            lr = 0.01

        # ----- TRAIN -----
        model.train()
        # ----- batch_sizeごとに処理 -----
        for x_train_batch, t_train_batch in dataloader_train:
            model.zero_grad() # 勾配の初期化
            y_train_batch = model.forward(x_train_batch)
            loss_train_batch = criterion(y_train_batch, t_train_batch)
            loss_train_batch.backward()
            optimizer.step()
            loss_train_batch_list.append(loss_train_batch.tolist())

        # ----- TEST -----
        model.eval()
        # ----- batch_sizeごとに処理 -----
        for x_test_batch, t_test_batch in dataloader_test:
            y_test_batch = model.forward(x_test_batch)
            loss_test_batch = criterion(y_test_batch, t_test_batch)
            loss_test_batch_list.append(loss_test_batch.tolist())

            label_test_batch = y_test_batch.argmax(1).numpy()
            t_test_batch = t_test_batch.numpy()
            correct += np.sum((label_test_batch - t_test_batch) == 0)
            total += 1

        loss_train_list.append(np.mean(loss_train_batch_list))
        loss_test_list.append(np.mean(loss_test_batch_list))
        acc_test_batch = 100 * correct / total
        acc_test_list.append(acc_test_batch)

        print('EPOCH: {}, lr: {:.2f},  TRAIN LOSS: {:.3f}, TEST LOSS: {:.3f}, TEST ACC: {:.3f}'.format(
            epoch,
            lr,
            np.mean(loss_train_list),
            np.mean(loss_test_list),
            acc_test_batch
        ))

        loss_before, stop_count = early_stopping.item(np.mean(loss_train_list), loss_before, stop_count)
        if early_stopping.early_stop(stop_count):
            print('Early Stopping')
            break

    # ----- PLOT -----
    plt.plot(loss_train_list)
    plt.show()
    plt.plot(loss_test_list)
    plt.show()
    plt.plot(acc_test_list)
    plt.show()

if __name__ == '__main__':
    mnist_learning(fashion=False, n_epoch=50, lr=0.1)