import torch

def do(logger, dataloader, model, loss_fn, optimizer, device):

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0 or batch == size:
            loss, current = loss.item(), batch * len(X)
            logger.info(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # calc train accuracy for this epoch
    correct = 0
    total = 0
    _, predicted = torch.max(pred.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum().item()

    # print for monitor
    print(f"train loss: {loss:>7f} train acc: {100 * correct / total:>3f}  [{size:>5d}/{size:>5d}]")
    logger.info(f"train loss: {loss:>7f} train acc: {100 * correct / total:>3f}  [{size:>5d}/{size:>5d}]")

    return loss, 100 * correct / total