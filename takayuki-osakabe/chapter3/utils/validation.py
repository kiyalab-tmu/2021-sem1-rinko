import torch
import torch.nn as nn

def validation(model, loader, criterion):
    model.eval()
    validation_loss, validation_acc = 0.0, 0.0

    for X, y in loader:
        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y)
            validation_acc += (outputs.max(dim=1)[1] == y).sum().item()
            validation_loss += loss.item() * X.shape[0]

    return validation_acc / len(loader.dataset), validation_loss / len(loader.dataset)
