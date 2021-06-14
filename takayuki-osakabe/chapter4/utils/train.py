import torch
import torch.nn as nn

def train(model, loader, opt, criterion, epoch, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for i, (X,y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        loss = criterion(outputs, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_acc += (outputs.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]

        if (i+1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(epoch+1,i*len(X), len(loader.dataset), 100.*i/len(loader), loss.data.item()))

    print('Training set: Accuracy: {}/{} ({:.0f}%)'.format(total_acc, len(loader.dataset), 100.*total_acc/len(loader.dataset)))

    return total_acc / len(loader.dataset), total_loss / len(loader.dataset)
