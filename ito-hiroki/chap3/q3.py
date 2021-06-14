import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def weights_init(m):
    if isinstance(m, nn.Linear):
        print("initialized!")
        nn.init.normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def one_epoch(model, data_loader, criterion, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    losses = 0
    data_num = 0
    correct_num = 0
    iter_num = 0

    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)
        data_num += len(targets)
        iter_num += 1

        if optimizer:
            logits = model(images)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, targets)

        losses += loss.item()

        prediction = torch.argmax(logits, dim=1)
        correct_num += (prediction == targets).sum().item()

    return losses / iter_num, correct_num / data_num


# ref: https://github.com/lessw2020/mish/blob/master/mish.py
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU)
        # vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


BATCH_SIZE = 256
EPOCH_NUM = 100
device = "cuda:0" if torch.cuda.is_available() else "cpu"
act_dict = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "leakey": nn.LeakyReLU(),
    "swish": nn.SiLU(),
    "mish": Mish(),
}

for act_name, activation in act_dict.items():
    print(f"Activation: {act_name}")
    # Reproducibility
    torch.manual_seed(100)
    random.seed(200)
    np.random.seed(300)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform = transforms.Compose([transforms.ToTensor()])

    all_train_ds = datasets.FashionMNIST(
        "../data/", train=True, transform=transform, download=False
    )
    train_idx, valid_idx = train_test_split(
        np.arange(len(all_train_ds.targets)),
        test_size=0.1,
        shuffle=True,
        stratify=all_train_ds.targets,
        random_state=19980307,
    )
    train_ds = torch.utils.data.Subset(all_train_ds, train_idx)
    valid_ds = torch.utils.data.Subset(all_train_ds, valid_idx)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        activation,
        nn.Linear(256, 10),
    )
    model.apply(weights_init)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    best_valid_loss, best_valid_accuracy = None, None
    for epoch in tqdm(range(EPOCH_NUM)):
        loss, acc = one_epoch(model, train_dl, criterion, optimizer)
        train_loss.append(loss)
        train_acc.append(acc)

        loss, acc = one_epoch(model, valid_dl, criterion)
        valid_loss.append(loss)
        valid_acc.append(acc)

        if epoch == 0 or best_valid_loss >= loss:
            best_valid_loss = loss
            best_valid_accuracy = acc
            torch.save(model.state_dict(), f"best_checkpoint_{act_name}.pth")

    print(f"best valid loss: {best_valid_loss:.3}, accuracy: {best_valid_accuracy:.3%}")

    model.load_state_dict(torch.load(f"best_checkpoint_{act_name}.pth"))
    test_ds = datasets.FashionMNIST(
        "../data/", train=False, transform=transform, download=False
    )
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loss, test_acc = one_epoch(model, test_dl, criterion)
    print(f"Test Loss: {test_loss:.3}, Accuracy: {test_acc:.2%}")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(train_loss, color="limegreen", label="Train Loss")
    ax1.plot(valid_loss, color="purple", label="Valid Loss")
    ax2 = ax1.twinx()
    ax2.plot(train_acc, color="skyblue", label="Train Acc")
    ax2.plot(valid_acc, color="pink", label="Valid Acc")

    ax1.set_title(f"Learning Curve (Test Loss: {test_loss:.3}, Acc: {test_acc:.2%})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Acc")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")

    fig.savefig(f"{act_name}.png")
