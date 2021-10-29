import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale, Normalize
import collections
import numpy as np
import math

class BasicDatasetCIFAR10:

    def __init__(self, batch_size, path, transforms, isVALID=-1, isReduceTrain=-1):
        # Download training data from open datasets.
        training_data = datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=transforms
        )

        # Download test data from open datasets.
        test_data = datasets.CIFAR10(
            root=path,
            train=False,
            download=True,
            transform=transforms
        )

        # Create data loaders.
        # Reduce train dataset for some reasons.
        self.num_train = len(training_data)
        self.num_valid = 0
        if isReduceTrain != -1:
            training_data, _ = torch.utils.data.random_split(training_data, 
                               [isReduceTrain, len(training_data)-isReduceTrain])
            self.num_train = len(training_data)
        # for validation.
        if isVALID != -1:
            self.train, self.valid = torch.utils.data.random_split(training_data, 
                                     [len(training_data)-isVALID, isVALID])
            self.train = DataLoader(self.train, batch_size=batch_size)
            self.valid = DataLoader(self.valid, batch_size=batch_size)
            self.num_valid = isVALID
        else:
            self.train = DataLoader(training_data, batch_size=batch_size)
        
        self.test  = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        

    def count_test_dataset_num_class(self, test_data):
        y_all = []
        for _, y in test_data:
            y_all.append(y)
        y_count = collections.Counter(y_all)
        self.num_per_class = y_count.most_common() #[(label_id, num), (9, 1000), (2, 1000)... (5,1000)] NOT in oder
        self.num_class     = len(self.num_per_class) # num class

    def add_class_label(self, labels):
        self.labels = labels

class BasicDatasetMNIST:

    def __init__(self, batch_size, path, transforms, isVALID=-1, isReduceTrain=-1):
        # Download training data from open datasets.
        training_data = datasets.MNIST(
            root=path,
            train=True,
            download=True,
            transform=transforms
        )

        # Download test data from open datasets.
        test_data = datasets.MNIST(
            root=path,
            train=False,
            download=True,
            transform=transforms
        )

        # Create data loaders.
        # Reduce train dataset for some reasons.
        self.num_train = len(training_data)
        self.num_valid = 0
        if isReduceTrain != -1:
            training_data, _ = torch.utils.data.random_split(training_data, 
                               [isReduceTrain, len(training_data)-isReduceTrain])
            self.num_train = len(training_data)
        # for validation.
        if isVALID != -1:
            self.train, self.valid = torch.utils.data.random_split(training_data, 
                                     [len(training_data)-isVALID, isVALID])
            self.train = DataLoader(self.train, batch_size=batch_size)
            self.valid = DataLoader(self.valid, batch_size=batch_size)
            self.num_valid = isVALID
        else:
            self.train = DataLoader(training_data, batch_size=batch_size)
        
        self.test  = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        

    def count_test_dataset_num_class(self, test_data):
        y_all = []
        for _, y in test_data:
            y_all.append(y)
        y_count = collections.Counter(y_all)
        self.num_per_class = y_count.most_common() #[(label_id, num), (9, 1000), (2, 1000)... (5,1000)] NOT in oder
        self.num_class     = len(self.num_per_class) # num class

    def add_class_label(self, labels):
        self.labels = labels

class BasicDataset:

    def __init__(self, batch_size, path, isVALID=-1, isReduceTrain=-1):
        # Download training data from open datasets.
        training_data = datasets.MNIST(
            root=path,
            train=True,
            download=False,
            transform=Compose([
                Resize(64),
                ToTensor(),
                Normalize([0.5], [0.5])
            ])
        )

        # Download test data from open datasets.
        test_data = datasets.MNIST(
            root=path,
            train=False,
            download=False,
            transform=Compose([
                Resize(64),
                ToTensor(),
                Normalize([0.5], [0.5])
            ])
        )

        # Create data loaders.
        # Reduce train dataset for some reasons.
        self.num_train = len(training_data)
        self.num_valid = 0
        if isReduceTrain != -1:
            training_data, _ = torch.utils.data.random_split(training_data, 
                               [isReduceTrain, len(training_data)-isReduceTrain])
            self.num_train = len(training_data)
        # for validation.
        if isVALID != -1:
            self.train, self.valid = torch.utils.data.random_split(training_data, 
                                     [len(training_data)-isVALID, isVALID])
            self.train = DataLoader(self.train, batch_size=batch_size)
            self.valid = DataLoader(self.valid, batch_size=batch_size)
            self.num_valid = isVALID
        else:
            self.train = DataLoader(training_data, batch_size=batch_size)
        
        self.test  = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        

    def count_test_dataset_num_class(self, test_data):
        y_all = []
        for _, y in test_data:
            y_all.append(y)
        y_count = collections.Counter(y_all)
        self.num_per_class = y_count.most_common() #[(label_id, num), (9, 1000), (2, 1000)... (5,1000)] NOT in oder
        self.num_class     = len(self.num_per_class) # num class

    def add_class_label(self, labels):
        self.labels = labels

class BasicDatasetTo3ch:

    def __init__(self, batch_size, path, isVALID=-1, isReduceTrain=-1):
        # Download training data from open datasets.
        training_data = datasets.MNIST(
            root=path,
            train=True,
            download=False,
            transform=Compose([
                Grayscale(num_output_channels=3),
                Resize(64),
                ToTensor(),
                Normalize([0.5], [0.5])
            ])
        )

        # Create data loaders.
        # Reduce train dataset for some reasons.
        self.num_train = len(training_data)
        self.num_valid = 0
        if isReduceTrain != -1:
            training_data, _ = torch.utils.data.random_split(training_data, 
                               [isReduceTrain, len(training_data)-isReduceTrain])
            self.num_train = len(training_data)
        # for validation.
        if isVALID != -1:
            self.train, self.valid = torch.utils.data.random_split(training_data, 
                                     [len(training_data)-isVALID, isVALID])
            self.train = DataLoader(self.train, batch_size=batch_size)
            self.valid = DataLoader(self.valid, batch_size=batch_size)
            self.num_valid = isVALID
        else:
            self.train = DataLoader(training_data, batch_size=batch_size)
        

    def count_test_dataset_num_class(self, test_data):
        y_all = []
        for _, y in test_data:
            y_all.append(y)
        y_count = collections.Counter(y_all)
        self.num_per_class = y_count.most_common() #[(label_id, num), (9, 1000), (2, 1000)... (5,1000)] NOT in oder
        self.num_class     = len(self.num_per_class) # num class

    def add_class_label(self, labels):
        self.labels = labels

class BasicDatasetForVAE:

    def __init__(self, batch_size, path, isVALID=-1, isReduceTrain=-1):
        # Download training data from open datasets.
        training_data = datasets.MNIST(
            root=path,
            train=True,
            download=False,
            transform=Compose([
                ToTensor(),
                Normalize([0], [1])
            ])
        )

        # Download test data from open datasets.
        test_data = datasets.MNIST(
            root=path,
            train=False,
            download=False,
            transform=ToTensor(),
        )

        # Create data loaders.
        # Reduce train dataset for some reasons.
        self.num_train = len(training_data)
        self.num_valid = 0
        if isReduceTrain != -1:
            training_data, _ = torch.utils.data.random_split(training_data, 
                               [isReduceTrain, len(training_data)-isReduceTrain])
            self.num_train = len(training_data)
        # for validation.
        if isVALID != -1:
            self.train, self.valid = torch.utils.data.random_split(training_data, 
                                     [len(training_data)-isVALID, isVALID])
            self.train = DataLoader(self.train, batch_size=batch_size)
            self.valid = DataLoader(self.valid, batch_size=batch_size)
            self.num_valid = isVALID
        else:
            self.train = DataLoader(training_data, batch_size=batch_size)
        
        self.test  = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        

    def count_test_dataset_num_class(self, test_data):
        y_all = []
        for _, y in test_data:
            y_all.append(y)
        y_count = collections.Counter(y_all)
        self.num_per_class = y_count.most_common() #[(label_id, num), (9, 1000), (2, 1000)... (5,1000)] NOT in oder
        self.num_class     = len(self.num_per_class) # num class

    def add_class_label(self, labels):
        self.labels = labels


class BasicDatasetForGAN:

    def __init__(self, batch_size, path, isVALID=-1, isReduceTrain=-1):
        # Download training data from open datasets.
        training_data = datasets.MNIST(
            root=path,
            train=True,
            download=False,
            transform=Compose([
                ToTensor(),
                Normalize([0], [1])
            ])
        )

        # Download test data from open datasets.
        test_data = datasets.MNIST(
            root=path,
            train=False,
            download=False,
            transform=ToTensor(),
        )

        # Create data loaders.
        # Reduce train dataset for some reasons.
        self.num_train = len(training_data)
        self.num_valid = 0
        if isReduceTrain != -1:
            training_data, _ = torch.utils.data.random_split(training_data, 
                               [isReduceTrain, len(training_data)-isReduceTrain])
            self.num_train = len(training_data)
        # for validation.
        if isVALID != -1:
            self.train, self.valid = torch.utils.data.random_split(training_data, 
                                     [len(training_data)-isVALID, isVALID])
            self.train = DataLoader(self.train, batch_size=batch_size)
            self.valid = DataLoader(self.valid, batch_size=batch_size)
            self.num_valid = isVALID
        else:
            self.train = DataLoader(training_data, batch_size=batch_size)
        
        self.test  = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        

    def count_test_dataset_num_class(self, test_data):
        y_all = []
        for _, y in test_data:
            y_all.append(y)
        y_count = collections.Counter(y_all)
        self.num_per_class = y_count.most_common() #[(label_id, num), (9, 1000), (2, 1000)... (5,1000)] NOT in oder
        self.num_class     = len(self.num_per_class) # num class

    def add_class_label(self, labels):
        self.labels = labels

# used in 6-a
class BasicDataset_Gaussian_Mixture_Double_Circle(Dataset):

    def __init__(self, num_data, transform=None, num_cluster=8, scale=1, std=1):
        self.num_data = num_data
        self.transform = transform
        self.data = self.gaussian_mixture_double_circle(self.num_data, num_cluster, scale, std)

    def gaussian_mixture_double_circle(self, batchsize, num_cluster=8, scale=2, std=0.2):
        rand_indices = np.random.randint(0, num_cluster, size=batchsize)
        base_angle = math.pi * 2 / num_cluster
        angle = rand_indices * base_angle - math.pi / 2
        mean = np.zeros((batchsize, 2), dtype=np.float32)
        mean[:, 0] = np.cos(angle) * scale
        mean[:, 1] = np.sin(angle) * scale
        # Doubles the scale in case of even number
        even_indices = np.argwhere(rand_indices % 2 == 0)
        mean[even_indices] /= 2
        return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return torch.Tensor((self.data[idx]))