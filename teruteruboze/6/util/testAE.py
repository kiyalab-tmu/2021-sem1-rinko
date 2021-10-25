import torch
import numpy as np
def do(logger, dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, _ in dataloader:
            X = X.view(X.size(0), -1)
            X = X.to(device)
            out, _ = model(X)
            test_loss += loss_fn(out, X).item()

    test_loss /= size
    logger.info(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return None