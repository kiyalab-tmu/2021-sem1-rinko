import torch

def do(logger, dataloader, model, loss_fn, optimizer, device):

    size = len(dataloader.dataset)
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            if batch % 100 == 0 or batch == size:
                loss, current = loss.item(), batch * len(X)
                logger.info(f"valid loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # calc valid accuracy for this epoch
        correct = 0
        total = 0
        _, predicted = torch.max(pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        # print for monitor
        print(f"valid loss: {loss:>7f} valid acc: {100 * correct / total:>3f}  [{size:>5d}/{size:>5d}]")
        logger.info(f"valid loss: {loss:>7f} valid acc: {100 * correct / total:>3f}  [{size:>5d}/{size:>5d}]")

    return loss, 100 * correct / total
    