import numpy as np
import pandas as pd
import torch
from torchtext.legacy.data import Dataset


class TimeMachineDataset(Dataset):
    def __init__(
        self,
        text_path="../data/TheTimeMachine/35-0.txt",
        dict_path="../data/TheTimeMachine/char.csv",
    ):
        self.text_path = text_path
        with open(text_path, "r", encoding="utf-8-sig") as file:
            self.data = file.readlines()
        self.map_dict = pd.read_csv(dict_path)
        self.vocab_size = len(self.map_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = list(self.data[idx].strip())
        text = [self.map_dict(x) for x in text]
        return text


class TimeMachineData:
    def __init__(
        self,
        text_path="../data/TheTimeMachine/35-0.txt",
        dict_path="../data/TheTimeMachine/char.csv",
    ):
        self.text_path = text_path
        with open(text_path, "r", encoding="utf-8-sig") as file:
            self.data = file.readlines()
        word_df = pd.read_csv(dict_path)
        self.map_dict = dict(zip(word_df["word"], word_df.index))
        self.vocab_size = len(self.map_dict)

    def get_data(self):
        data = []
        for text in self.data:
            if text.strip() != "":
                text = (
                    [self.map_dict["<BOS>"]]
                    + [self.map_dict[char] for char in text.strip()]
                    + [self.map_dict["<EOS>"]]
                )
                data.append(text)

        data = torch.from_numpy(np.array(sum(data, []))).long()
        return data
