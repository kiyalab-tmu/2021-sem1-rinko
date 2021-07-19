import argparse
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from dataset import TimeMachineData, TimeMachineDataset
from model import GRU, RNN, LSTM


def worker_init_fn(worker_id):
    random.seed(worker_id)


def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    # Constants
    bptt = 35
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    EPOCH_NUM = 100
    CHECKPOINT_FOLDER = "./checkpoints/"
    NUM_WORKER = 2
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    save_name = f"{args.model}_time_machine.pth"

    if not os.path.exists(CHECKPOINT_FOLDER):
        os.makedirs(CHECKPOINT_FOLDER)

    # Reproducibility
    torch.manual_seed(100)
    random.seed(200)
    np.random.seed(300)
    cudnn.deterministic = True
    cudnn.benchmark = False

    timemachine = TimeMachineData()
    data = timemachine.get_data()

    ntokens = timemachine.vocab_size
    div_idx = int(len(data) * 0.8)
    train_data = batchify(data[:div_idx], TRAIN_BATCH_SIZE)
    test_data = batchify(data[div_idx:], TEST_BATCH_SIZE)

    if args.model == "rnn":
        model = RNN(ntokens).to(device)
    elif args.model == "gru":
        model = GRU(ntokens).to(device)
    elif args.model == "lstm":
        model = LSTM(ntokens).to(device)
    else:
        raise ValueError("Invalid model argument: {}".format(args.model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40, 60, 80], gamma=0.5
    )

    def train():
        model.train()  # Turn on the train mode
        total_loss = 0.0
        start_time = time.time()
        hidden = model.init_hidden(TRAIN_BATCH_SIZE)
        if args.model == "lstm":
            cell = model.init_hidden(TRAIN_BATCH_SIZE)
        # hidden = None
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            optimizer.zero_grad()
            if args.model == "lstm":
                output, hidden, cell = model(data, hidden, cell)
            else:
                output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 10
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches | "
                    "lr {:02.2f} | ms/batch {:5.2f} | "
                    "loss {:5.2f} | ppl {:8.2f}".format(
                        epoch,
                        batch,
                        len(train_data) // bptt,
                        scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss,
                        math.exp(cur_loss),
                    )
                )
                start_time = time.time()
            hidden = repackage_hidden(hidden)
            if args.model == "lstm":
                cell = repackage_hidden(cell)
        return total_loss / (batch + 1)

    def evaluate(model, data_source):
        model.eval()  # Turn on the evaluation mode
        total_loss = 0.0
        hidden = model.init_hidden(TEST_BATCH_SIZE)
        if args.model == "lstm":
            cell = model.init_hidden(TRAIN_BATCH_SIZE)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                if args.model == "lstm":
                    output, hidden, cell = model(data, hidden, cell)
                else:
                    output, hidden = model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
                hidden = repackage_hidden(hidden)
                if args.model == "lstm":
                    cell = repackage_hidden(cell)
        return total_loss / (len(data_source) - 1)

    best_val_loss = float("inf")
    best_model = None

    train_perp_list = []
    test_perp_list = []
    for epoch in range(EPOCH_NUM):
        epoch_start_time = time.time()
        trn_loss = train()
        train_perp_list.append(math.exp(trn_loss))
        val_loss = evaluate(model, test_data)
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
            "valid ppl {:8.2f}".format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
            )
        )
        test_perp_list.append(math.exp(val_loss))
        print("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
        torch.save(model.state_dict(), f"{args.model}_thetimemachine.pth")

        scheduler.step()

    plt.plot(train_perp_list, label="train")
    plt.plot(test_perp_list, label="test")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("perplexity")
    plt.savefig(f"{args.model}_thetimemachine_perplexity.png")
