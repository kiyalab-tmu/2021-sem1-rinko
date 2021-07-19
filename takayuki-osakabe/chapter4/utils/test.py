import torch
import torch.nn as nn

def test(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for X,y in loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_acc += (outputs.max(dim=1)[1] == y).sum().item()
            total_loss += loss.item() * X.shape[0]

    return total_acc / len(loader.dataset), total_loss / len(loader.dataset)
