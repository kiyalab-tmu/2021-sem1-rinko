import torch
import torch.nn as nn

def train(model, loader, opt, criterion, epoch):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for X,y in loader:
        outputs = model(X)
        loss = criterion(outputs, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_acc += (outputs.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]

    if (epoch+1) % 10 == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch+1, loss.data.item()))

    return total_acc / len(loader.dataset), total_loss / len(loader.dataset)
