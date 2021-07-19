import re
import copy
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import random


num_chars = 40   # kind of characters
batch_size = 128
partition = 20
hidden_size = 40
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def character_tokenizer(lines):
    char_dict = {}   # {"character": num of the character} : no diplicates
    # create new_lines
    new_lines = []   # new_line[n] keep all character which appear in lines[n]
    text_cleaner = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋—￥％…\‘\’]')
    for line in lines:
        line = text_cleaner.sub('', line).lower()
        chars = []
        for c in line:
            chars
            if c ==' ' or c=='\n':
                continue
            if c in char_dict:
                char_dict[c] += 1
            else:
                char_dict[c] = 1
            chars.append(c)
        new_lines.append(chars)
    
    # soted items(characters) according to its frequency
    soted_char_list = sorted(char_dict.items(), key=lambda x:x[1])
    soted_char_list.reverse()

    all_char = []   # keep all word in text according to its frequency
    for i in range(len(soted_char_list)):
        all_char.append(soted_char_list[i][0])
    
    # replace words with its index in new_lines
    for i in range(len(new_lines)):
        for j in range(len(new_lines[i])):
            new_lines[i][j] = all_char.index(new_lines[i][j])+1
    
    return new_lines


class TTMDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True, partition=20):
        super().__init__()
        DataPath = 'data/TheTimeMachine.txt'
        # DataPath = 'jimar884/chapter5/TheTimeMachine.txt'
        file = open(DataPath, 'r', encoding='utf-8-sig')   # utf-8-sig removes '/ufeff'
        lines = file.readlines()
        lines = character_tokenizer(lines)
        random.shuffle(lines)
        train = lines[:int(len(lines)*0.9)]
        test = lines[int(len(lines)*0.9):]
        self.data = []
        self.label = []
        if is_train:
            self.data, self.label = self._set_DataAndLabel(train, partition)
        else:
            self.data, self.label = self._set_DataAndLabel(test, partition)

    def _set_DataAndLabel(self, lines, partition):
        data, label = [], []
        for line in lines:
            if len(line) < partition:
                continue
            for i in range(len(line) - partition - 1):
                x = torch.tensor(line[i:i+partition])
                data.append(x)
                y = torch.tensor(line[i + partition])
                label.append(y)
        return data, label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size+hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size+output_size, output_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        return output, hidden


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1)).float()
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        resetgate = self.sigmoid(i_r + h_r)
        inputgate = self.sigmoid(i_i + h_i)
        newgate = self.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        self.hn = Variable(torch.zeros(batch_size, self.hidden_dim).to(device))
       
        self.gru_cell = GRUCell(input_dim, hidden_dim, bias)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
     
    
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        
        self.hn = self.gru_cell(x, self.hn)
        out = self.hn
        
        out = self.fc(out) 
        # out.size() --> 100, 10
        return out


def main():
    # create dataset and dataloader
    train_dataset = TTMDataset(is_train=True, partition=partition)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = TTMDataset(is_train=False, partition=partition)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    # define net, cirterion and optimizer
    net = GRUModel(partition, hidden_size, num_chars).to(device)
    criterion = nn.CrossEntropyLoss()
    tr_loss = []
    ts_loss = []
    tr_perplexity = []
    ts_perplexity = []
    for epoch in range(50):
        epoch_loss = .0
        epoch_perplexity = .0
        for inputs, labels in train_dataloader:
            if len(inputs) != batch_size:# batch size の数が最後合わないから無視
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            net.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            perplexity = torch.exp(loss)
            # perplexity = 2**loss.item()
            for p in net.parameters():
                p.data.add_(p.grad.data, alpha=-0.0005)
            epoch_loss += loss.item() * inputs.size(0)
            epoch_perplexity += perplexity.item() * inputs.size(0)
            # epoch_perplexity += perplexity * inputs.size(0)

        epoch_loss = epoch_loss / len(train_dataloader.dataset)
        epoch_perplexity  = epoch_perplexity / len(train_dataloader.dataset)
        print(epoch+1, ":", epoch_loss, epoch_perplexity)
        tr_loss.append(epoch_loss)
        tr_perplexity.append(epoch_perplexity)
        
        with torch.no_grad():
            epoch_loss = .0
            epoch_perplexity = .0
            for inputs, labels in test_dataloader:
                if len(inputs) != batch_size:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                perplexity = torch.exp(loss)
                # perplexity = 2**loss.item()
                epoch_loss += loss.item() * inputs.size(0)
                epoch_perplexity += perplexity.item() * inputs.size(0)
                # epoch_perplexity += perplexity * inputs.size(0)
            epoch_loss = epoch_loss / len(test_dataloader.dataset)
            epoch_perplexity = epoch_perplexity / len(test_dataloader.dataset)
            ts_loss.append(epoch_loss)
            ts_perplexity.append(epoch_perplexity)
    print(tr_loss)
    print(ts_loss)
    print(tr_perplexity)
    print(ts_perplexity)



if __name__ == "__main__":
    main()