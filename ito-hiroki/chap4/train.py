import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from matplotlib import pyplot as plt

from model import VGG11, AlexNet, LeNet, SmallAlexNet, SmallVGG11, LeNetBN
from densenet import densenet121
from network_in_network import NiN
from inception import Inception
from resnet import resnet20


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
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 128
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
    cudnn.benchmark = True

    if args.dataset == "fashion":
        num_classes = 10

        if args.model in ["lenet", "lenetbn", "resnet20"]:
            transform_train = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(28, padding=4),
                    transforms.ToTensor(),
                ]
            )
            transform_test = transforms.Compose([transforms.ToTensor()])
        elif args.model in ["alexnet", "vgg11", "densenet121", "inception", "nin"]:
            transform_train = transforms.Compose(
                [
                    transforms.Resize(224),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(224, padding=32),
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
    elif args.model == "densenet121":
        model = densenet121().to(device)
    elif args.model == "nin":
        model = NiN().to(device)
    elif args.model == "inception":
        model = Inception().to(device)
    elif args.model == "lenetbn":
        model = LeNetBN().to(device)
    elif args.model == "resnet20":
        model = resnet20().to(device)
    else:
        raise ValueError(f"model argment is invalid. {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005
    )
    # for
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    optim_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[25, 50, 75], gamma=0.2
    )

    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    for epoch in range(100):
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

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(train_loss_list, label="train_loss", color="red")
    ax[0].plot(test_loss_list, label="test_loss", color="blue")
    ax[1].plot(train_accuracy_list, label="train_acc", color="red")
    ax[1].plot(test_accuracy_list, label="test_acc", color="blue")
    fig.savefig(f"baseline_{args.model}_{args.dataset}.png")
