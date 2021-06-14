import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset

def load_fashionmnist(valid_size=0.2, batch_size=256, random_seed=416):
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    train_val_dataset = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True, transform=transform)

    num_train = len(train_val_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    
    train_dataset = Subset(train_val_dataset, train_idx)
    val_dataset = Subset(train_val_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
