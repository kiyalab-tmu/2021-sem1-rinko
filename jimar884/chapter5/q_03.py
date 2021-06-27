import re
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random


num_chars = 40   # kind of characters
batch_size = 256
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
    net = RNN(input_size=partition, hidden_size=hidden_size, output_size=num_chars).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # train
    
    hiddens = torch.zeros(batch_size, hidden_size).to(device)
    tr_loss = []
    ts_loss = []
    tr_acc = []
    ts_acc = []
    for epoch in range(200):
        epoch_loss = .0
        correct = 0
        total = 0
        for inputs, labels in train_dataloader:
            if len(inputs) != len(hiddens):# batch size の数が最後合わないから無視
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            # optimizer.zero_grad()
            net.zero_grad()
            outputs, hiddens = net(inputs, hiddens)
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            # optimizer.step()
            for p in net.parameters():
                p.data.add_(p.grad.data, alpha=-0.0005)
            epoch_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted==labels).sum().item()
        epoch_loss = epoch_loss / len(train_dataloader.dataset)
        current_accuracy = correct / total
        # print("train::", epoch_loss, "::", current_accuracy)
        print(epoch+1, ":", epoch_loss)
        tr_loss.append(epoch_loss)
        tr_acc.append(current_accuracy)
        
        with torch.no_grad():
            epoch_loss = .0
            correct = 0
            total = 0
            hiddens = torch.zeros(batch_size, hidden_size).to(device)
            for inputs, labels in test_dataloader:
                if len(inputs) != len(hiddens):
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, hiddens = net(inputs, hiddens)
                loss = criterion(outputs, labels)
                epoch_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += len(labels)
                correct += (predicted==labels).sum().item()
            epoch_loss = epoch_loss / len(test_dataloader.dataset)
            current_accuracy = correct / total
            # print("train::", epoch_loss, "::", current_accuracy)
            ts_loss.append(epoch_loss)
            ts_acc.append(current_accuracy)
            # print("acc in epoch{0} : {1}".format(epoch+1, current_accuracy))
    print(tr_loss)
    print(ts_loss)
    print(tr_acc)
    print(ts_acc)



if __name__ == "__main__":
    main()