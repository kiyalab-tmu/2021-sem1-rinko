import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SoftmaxRegression


def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8))


BATCH_SIZE = 256
EPOCH_NUM = 100

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ]
)

train_ds = datasets.FashionMNIST(
    "../data/", train=True, transform=transform, download=True
)
test_ds = datasets.FashionMNIST(
    "../data/", train=False, transform=transform, download=True
)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = SoftmaxRegression()
for epoch in range(EPOCH_NUM):
    for X, y in train_dl:
        y = torch.nn.functional.one_hot(y, num_classes=10)
        X, y = X.numpy(), y.numpy()
        model.update(X, y)

    loss = 0
    acc = 0
    for X, y in test_dl:
        y = torch.nn.functional.one_hot(y, num_classes=10)
        X, y = X.numpy(), y.numpy()
        y_pred = model.predict(X)
        loss += cross_entropy_loss(y, y_pred)
        acc += (np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)).sum()
    loss /= len(test_dl)
    acc /= len(test_ds)
    print(f"EPOCH {epoch}:, cel: {loss}, acc: {acc}")
