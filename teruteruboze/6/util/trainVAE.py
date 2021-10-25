import torch
import torch.nn as nn


def do(logger, dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device)

        x_hat, mu, logvar, _ = model(X)
        loss = loss_fn(x_hat, X, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        if batch % 100 == 0 or batch == size:
            current = batch * len(X)
            logger.info(f"train loss: {train_loss/size:>7f}  [{current:>5d}/{size:>5d}]")

    # print for monitor
    print(f"train loss: {train_loss/size:>7f}  [{size:>5d}/{size:>5d}]")
    logger.info(f"train loss: {train_loss/size:>7f}  [{size:>5d}/{size:>5d}]")

    return loss