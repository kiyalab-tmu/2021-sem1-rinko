import math

import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, nlayers=2):
        super(RNN, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.rnn = nn.RNN(emb_dim, hidden_dim, nlayers)
        self.dropout2 = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        nn.init.normal_(self.embed.weight, std=0.01)

        nn.init.normal_(self.rnn.weight_ih_l0, std=1 / math.sqrt(emb_dim))
        nn.init.normal_(self.rnn.weight_hh_l0, std=1 / math.sqrt(emb_dim))
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)

    def forward(self, inputs, hidden):
        out = self.embed(inputs)
        out = self.dropout1(out)
        out, hidden = self.rnn(out, hidden)
        out = self.dropout2(out)
        out = self.linear(out)

        return out, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.hidden_dim).zero_())


class GRU(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, nlayers=2):
        super(GRU, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.gru = nn.GRU(emb_dim, hidden_dim, nlayers)
        self.dropout2 = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        nn.init.normal_(self.embed.weight, std=0.01)

        nn.init.normal_(self.gru.weight_ih_l0, std=1 / math.sqrt(emb_dim))
        nn.init.normal_(self.gru.weight_hh_l0, std=1 / math.sqrt(emb_dim))
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

    def forward(self, inputs, hidden):
        out = self.embed(inputs)
        out = self.dropout1(out)
        out, hidden = self.gru(out, hidden)
        out = self.dropout2(out)
        out = self.linear(out)

        return out, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.hidden_dim).zero_())


class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, nlayers=2):
        super(LSTM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, nlayers, dropout=0.3)
        self.dropout2 = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim, vocab_size)

        nn.init.normal_(self.embed.weight, std=0.01)

        nn.init.normal_(self.lstm.weight_ih_l0, std=1 / math.sqrt(emb_dim))
        nn.init.normal_(self.lstm.weight_hh_l0, std=1 / math.sqrt(emb_dim))
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)

    def forward(self, inputs, hidden, cell):
        out = self.embed(inputs)
        out = self.dropout1(out)
        out, (hidden, cell) = self.lstm(out, (hidden, cell))
        out = self.dropout2(out)
        out = self.linear(out)

        return out, hidden, cell

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, bsz, self.hidden_dim).zero_())
