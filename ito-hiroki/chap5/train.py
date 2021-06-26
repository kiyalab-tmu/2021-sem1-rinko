import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable

from dataset import TimeMachineData, TimeMachineDataset


def worker_init_fn(worker_id):
    random.seed(worker_id)


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128):
        super(RNN, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.rnn = nn.RNN(emb_dim, hidden_dim, 2)
        self.dropout2 = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        nn.init.normal_(self.embed.weight, std=0.01)

        nn.init.normal_(self.rnn.weight_ih_l0, std=1 / math.sqrt(emb_dim))
        nn.init.normal_(self.rnn.weight_hh_l0, std=1 / math.sqrt(emb_dim))
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)

    def forward(self, inputs):
        out = self.embed(inputs)
        out = self.dropout1(out)
        out, _ = self.rnn(out)
        out = self.dropout2(out)
        out = self.linear(out)

        return out


def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


bptt = 35


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target


# def repackage_hidden(h):
#     """Wraps hidden states in new Variables, to detach them from their history."""
#     if type(h) == Variable:
#         return Variable(h.data)
#     else:
#         return tuple(repackage_hidden(v) for v in h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    # Constants
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 256
    EPOCH_NUM = 50
    CHECKPOINT_FOLDER = "./checkpoints/"
    NUM_WORKER = 2
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    save_name = f"rnn_time_machine.pth"

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
    batch_size = 20
    div_idx = int(len(data) * 0.8)
    train_data = batchify(data[:div_idx], batch_size)
    test_data = batchify(data[div_idx:], batch_size)

    model = RNN(ntokens).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.2)

    def train():
        model.train()  # Turn on the train mode
        total_loss = 0.0
        start_time = time.time()
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 200
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
                total_loss = 0
                start_time = time.time()

    def evaluate(eval_model, data_source):
        eval_model.eval()  # Turn on the evaluation mode
        total_loss = 0.0
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                output = eval_model(data)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(EPOCH_NUM):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, test_data)
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
            "valid ppl {:8.2f}".format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
            )
        )
        print("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()
