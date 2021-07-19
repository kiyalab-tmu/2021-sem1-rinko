import torch
import torch.nn as nn

def validation(model, loader, criterion, device):
    model.eval()
    validation_loss, validation_acc = 0.0, 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y)
            validation_acc += (outputs.max(dim=1)[1] == y).sum().item()
            validation_loss += loss.item() * X.shape[0]

    print('Validation set: Accuracy: {}/{} ({:.0f}%)'.format(validation_acc, len(loader.dataset), 100.*validation_acc/len(loader.dataset)))

    return validation_acc / len(loader.dataset), validation_loss / len(loader.dataset)
