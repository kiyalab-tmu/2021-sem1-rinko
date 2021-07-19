import torch
import numpy as np
def do(logger, dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    # for confusion matrix
    confusion_matrix_y_pred = np.array([])
    confusion_matrix_y_true = np.array([])
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).softmax(dim=1)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # for confusion matrix
            confusion_matrix_y_pred = np.append(confusion_matrix_y_pred, pred.argmax(1).to('cpu').detach().numpy().copy())
            confusion_matrix_y_true = np.append(confusion_matrix_y_true, y.to('cpu').detach().numpy().copy())

    test_loss /= size
    correct /= size
    logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return confusion_matrix_y_pred, confusion_matrix_y_true