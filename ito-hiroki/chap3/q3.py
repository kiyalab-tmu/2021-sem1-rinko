import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data)
        nn.init.normal_(m.bias.data)


def one_epoch(model, data_loader, criterion, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    losses = 0
    data_num = 0
    correct_num = 0
    iter_num = 0

    for images, targets in tqdm(data_loader):
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


BATCH_SIZE = 256
EPOCH_NUM = 100
device = "cuda:0" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([transforms.ToTensor()])

train_ds = datasets.FashionMNIST(
    "../data/", train=True, transform=transform, download=True
)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_ds = datasets.FashionMNIST(
    "../data/", train=False, transform=transform, download=True
)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

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

train_loss, train_acc = [], []
test_loss, test_acc = [], []
for epoch in range(EPOCH_NUM):
    loss, acc = one_epoch(model, train_dl, criterion, optimizer)
    train_loss.append(loss)
    train_acc.append(acc)

    loss, acc = one_epoch(model, train_dl, criterion)
    test_loss.append(loss)
    test_acc.append(acc)

print(f"Train Loss: {train_loss[-1]:.3}, Accuracy: {train_acc[-1]:%}")
print(f"Test Loss: {test_loss[-1]:.3}, Accuracy: {test_acc[-1]:%}")

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Learning Curve")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")

# Traing score と Test score をプロット
ax1.plot(train_loss, "o-", color="r", label="Train Loss")
ax1.plot(test_loss, "o-", color="g", label="Test Loss")
ax2 = ax1.twinx()
ax2.plot(train_acc, color="r", label="Train Acc")
ax2.plot(test_acc, color="g", label="Test Acc")
fig.savefig("img.png")