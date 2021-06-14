import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize
from tqdm import tqdm

from model import LeNet, AlexNet, SmallAlexNet, VGG11, SmallVGG11


def worker_init_fn(worker_id):
    random.seed(worker_id)


def one_epoch(model, data_loader, criterion, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    losses = 0
    data_num = 0
    correct_num = 0
    iter_num = 0

    for images, targets in tqdm(data_loader):
        images, targets = images.to(device), targets.to(device)
        data_num += len(targets)
        iter_num += 1

        if optimizer:
            logits = model(images)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, targets)

        losses += loss.item()

        prediction = torch.argmax(logits, dim=1)
        correct_num += (prediction == targets).sum().item()

    return losses / iter_num, correct_num / data_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="fashion")
    args = parser.parse_args()

    # Constants
    DATA_PATH = "../data/"
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 256
    EPOCH_NUM = 100
    CHECKPOINT_FOLDER = "./checkpoints/"
    NUM_WORKER = 2
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    save_name = f"baseline_{args.model}_{args.dataset}.pth"

    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    # Reproducibility
    torch.manual_seed(100)
    random.seed(200)
    np.random.seed(300)
    cudnn.deterministic = True
    cudnn.benchmark = False

    if args.dataset == "fashion":
        num_classes = 10

        if args.model == "lenet":
            transform_train = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(28, padding=4),
                    transforms.ToTensor(),
                ]
            )
            transform_test = transforms.Compose([transforms.ToTensor()])
        elif args.model in ["alexnet", "vgg11"]:
            transform_train = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(224, padding=32),
                    transforms.ToTensor(),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                ]
            )
        else:
            raise ValueError(f"model argment is invalid. {args.model}")

        train_ds = torchvision.datasets.FashionMNIST(
            root=DATA_PATH, train=True, download=True, transform=transform_train
        )
        test_ds = torchvision.datasets.FashionMNIST(
            root=DATA_PATH, train=False, download=True, transform=transform_test
        )

        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=NUM_WORKER,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        test_dl = torch.utils.data.DataLoader(
            test_ds,
            batch_size=TEST_BATCH_SIZE,
            num_workers=NUM_WORKER,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
    else:
        raise ValueError(f"dataset argment is invalid. {args.dataset}")

    if args.model == "lenet":
        model = LeNet().to(device)
    elif args.model == "alexnet":
        model = SmallAlexNet().to(device)
    elif args.model == "vgg11":
        model = SmallVGG11().to(device)
    else:
        raise ValueError(f"model argment is invalid. {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005
    )
    optim_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 50, 75], gamma=0.2
    )

    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    for epoch in range(EPOCH_NUM):
        print("EPOCH: {}".format(epoch))
        # train
        loss, accuracy = one_epoch(model, train_dl, criterion, optimizer=optimizer)
        train_loss_list.append(loss)
        train_accuracy_list.append(accuracy)
        print("train loss: {}, accuracy: {:.3%}".format(loss, accuracy))

        # test
        loss, accuracy = one_epoch(model, test_dl, criterion)
        test_loss_list.append(loss)
        test_accuracy_list.append(accuracy)
        print("test loss: {}, accuracy: {:.3%}".format(loss, accuracy))

        # step scheduler
        optim_scheduler.step()
        torch.save(model.state_dict(), CHECKPOINT_FOLDER + save_name)

    print(
        "last test loss: {:.3}, accuracy: {:.3%}".format(
            test_loss_list[-1], test_accuracy_list[-1]
        )
    )

    print("best test accuracy: {:.3%}".format(max(test_accuracy_list)))
