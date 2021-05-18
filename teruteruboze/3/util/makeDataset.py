import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose

class BasicDataset:

    def __init__(self, batch_size):
        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root="./3-2_Softmax_Regression/data",
            train=True,
            download=True,
            transform=Compose([
                ToTensor(),
                AddGaussianNoise(0., 1.)
            ])
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root="./3-2_Softmax_Regression/data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

        # Create data loaders.
        self.train = DataLoader(training_data, batch_size=batch_size)
        self.test  = DataLoader(test_data, batch_size=batch_size)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)