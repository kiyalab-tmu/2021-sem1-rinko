import torch

def do(logger, dataloader, model, loss_fn, optimizer, device):

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, _) in enumerate(dataloader):
        X = X.view(X.size(0), -1)
        X = X.to(device)

        out, _ = model(X)
        loss = loss_fn(X, out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0 or batch == size:
            loss, current = loss.item(), batch * len(X)
            logger.info(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # print for monitor
    print(f"train loss: {loss:>7f}  [{size:>5d}/{size:>5d}]")
    logger.info(f"train loss: {loss:>7f}  [{size:>5d}/{size:>5d}]")

    return loss