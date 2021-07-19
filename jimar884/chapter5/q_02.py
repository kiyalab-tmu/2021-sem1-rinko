import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


num_chars = 40   # kind of characters
batch_size = 256

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
        DataPath = 'jimar884/chapter5/TheTimeMachine.txt'
        file = open(DataPath, 'r', encoding='utf-8-sig')   # utf-8-sig removes '/ufeff'
        lines = file.readlines()
        lines = character_tokenizer(lines)
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


def main():
    train_dataset = TTMDataset(is_train=True, partition=20)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = TTMDataset(is_train=False, partition=20)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )


if __name__ == "__main__":
    main()