import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from q1 import Convert_charactor_to_index,Convert_word_to_index

class TMDataset(Dataset):
    def __init__(self,level="charactor",train = True,partition=30):
        super().__init__()
        if level == "charactor":
            sentences = Convert_charactor_to_index()
        elif level == "word":
            sentences = Convert_word_to_index()
        else:
            raise ValueError("Select 'charactor' or 'word' for then level")
        
        train_data = sentences[:int(len(sentences)*0.9)]
        test_data = sentences[int(len(sentences)*0.9):]
        
        self.data,self.label = [],[]
        
        if train:
            self.data,self.label = self._make_dataset(train_data,partition)
        else:
            self.data,self.label = self._make_dataset(test_data,partition)
        
    def _make_dataset(self,dataset,partition):
        data, label = [], []
        for line in dataset:
            if len(line) <= partition:
                continue
            for i in range(0, len(line) - partition - 1):
                x = torch.as_tensor(line[i:i+partition])
                x = x.reshape((1, -1))
                y = torch.as_tensor(line[i+partition+1])
                data.append(x)
                label.append(y)
                    
        return data, label
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

"""  
if __name__ == "__main__":
    train_dataset = TMDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataset = TMDataset(train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)
"""