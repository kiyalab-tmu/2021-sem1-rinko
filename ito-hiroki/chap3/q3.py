import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm, trange


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data)
        nn.init.normal_(m.bias.data)


BATCH_SIZE = 256
EPOCH_NUM = 100
device = "cuda:0" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: torch.flatten(x)),
    ]
)

train_ds = datasets.FashionMNIST(
    "../data/", train=True, transform=transform, download=True
)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
model.apply(weights_init)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_losses, train_accs = [], []
with trange(EPOCH_NUM) as pbar:
    for epoch in pbar:
        pbar.set_description(f"[Epoch {epoch}]")
        train_loss, train_acc = 0.0, 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_losses.append(train_loss / len(train_ds))
        train_accs.append(train_acc / len(train_ds))
print("Train Finish")
print(f"Loss: {train_losses[-1]:.3}, Accuracy: {train_accs[-1]:%}")


test_loss, test_acc = 0.0, 0.0
test_ds = datasets.FashionMNIST(
    "../data/", train=False, transform=transform, download=True
)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
for X, y in tqdm(test_dl):
    X, y = X.to(device), y.to(device)
    pred = model(X)
    loss = criterion(pred, y)
    test_loss += loss.item()
    test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
print("Test Finish")
print(f"Loss: {test_loss / len(test_ds):.3}, Accuracy: {test_acc / len(test_ds):%}")
