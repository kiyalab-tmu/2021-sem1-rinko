import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Grayscale
import collections

class BasicDataset:

    def __init__(self, batch_size, path, isVALID=-1, isReduceTrain=-1):
        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root=path,
            train=True,
            download=False,
            transform=Compose([
                ToTensor()
                #AddGaussianNoise(0., 1.)
            ])
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
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
        
        self.test  = DataLoader(test_data, batch_size=batch_size)
        

    def count_test_dataset_num_class(self, test_data):
        y_all = []
        for _, y in test_data:
            y_all.append(y)
        y_count = collections.Counter(y_all)
        self.num_per_class = y_count.most_common() #[(label_id, num), (9, 1000), (2, 1000)... (5,1000)] NOT in oder
        self.num_class     = len(self.num_per_class) # num class

    def add_class_label(self, labels):
        self.labels = labels

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class BasicDatasetPlus:

    def __init__(self, batch_size, path, isVALID=-1, resize=256, centercrop=224):
        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root=path,
            train=True,
            download=False,
            transform=Compose([
                Resize(resize),
                CenterCrop(centercrop),
                ToTensor()
            ])
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root=path,
            train=False,
            download=False,
            transform=Compose([
                Resize(resize),
                CenterCrop(centercrop),
                ToTensor()
            ])
        )

        # Create data loaders.
        if isVALID != -1:
            self.train, self.valid = torch.utils.data.random_split(training_data, 
                                     [len(training_data)-isVALID, isVALID])
            self.train = DataLoader(self.train, batch_size=batch_size)
            self.valid = DataLoader(self.valid, batch_size=batch_size)
        else:
            self.train = DataLoader(training_data, batch_size=batch_size)
        
        self.test  = DataLoader(test_data, batch_size=batch_size)
        

    def count_test_dataset_num_class(self, test_data):
        y_all = []
        for _, y in test_data:
            y_all.append(y)
        y_count = collections.Counter(y_all)
        self.num_per_class = y_count.most_common() #[(label_id, num), (9, 1000), (2, 1000)... (5,1000)] NOT in oder
        self.num_class     = len(self.num_per_class) # num class

    def add_class_label(self, labels):
        self.labels = labels

class BasicDataset_staticSplit:

    def __init__(self, batch_size, path, isVALID=-1, isReduceTrain=-1):
        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root=path,
            train=True,
            download=False,
            transform=Compose([
                ToTensor()
                #AddGaussianNoise(0., 1.)
            ])
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root=path,
            train=False,
            download=False,
            transform=ToTensor(),
        )

        # Create data loaders.
        # Reduce train dataset for some reasons.
        self.num_train = len(training_data)
        self.num_valid = 0
        # set random seed
        torch.manual_seed(192)
        if isReduceTrain != -1:
            training_data, _ = torch.utils.data.random_split(training_data, 
                               [isReduceTrain, len(training_data)-isReduceTrain])
            self.num_train = len(training_data)
        # for validation.
        if isVALID != -1:
            self.train, self.valid = torch.utils.data.random_split(training_data, 
                                     [len(training_data)-isVALID, isVALID])
            self.train = DataLoader(self.train, batch_size=batch_size, shuffle=True)
            self.valid = DataLoader(self.valid, batch_size=batch_size, shuffle=True)
            self.num_valid = isVALID
        else:
            self.train = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        
        self.test  = DataLoader(test_data, batch_size=batch_size)

        # reset seed
        torch.manual_seed(torch.initial_seed())
        

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
        training_data = datasets.FashionMNIST(
            root=path,
            train=True,
            download=False,
            transform=Compose([
                Grayscale(num_output_channels=3),
                ToTensor()
            ])
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root=path,
            train=False,
            download=False,
            transform=Compose([
                Grayscale(num_output_channels=3),
                ToTensor()
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
        
        self.test  = DataLoader(test_data, batch_size=batch_size)
        

    def count_test_dataset_num_class(self, test_data):
        y_all = []
        for _, y in test_data:
            y_all.append(y)
        y_count = collections.Counter(y_all)
        self.num_per_class = y_count.most_common() #[(label_id, num), (9, 1000), (2, 1000)... (5,1000)] NOT in oder
        self.num_class     = len(self.num_per_class) # num class

    def add_class_label(self, labels):
        self.labels = labels

class BasicDatasetTo3chEnlarge:

    def __init__(self, batch_size, path, isVALID=-1, isReduceTrain=-1, resize=256, centercrop=224):
        # Download training data from open datasets.
        training_data = datasets.FashionMNIST(
            root=path,
            train=True,
            download=False,
            transform=Compose([
                Grayscale(num_output_channels=3),
                Resize(resize),
                CenterCrop(centercrop),
                ToTensor()
            ])
        )

        # Download test data from open datasets.
        test_data = datasets.FashionMNIST(
            root=path,
            train=False,
            download=False,
            transform=Compose([
                Grayscale(num_output_channels=3),
                Resize(resize),
                CenterCrop(centercrop),
                ToTensor()
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
        
        self.test  = DataLoader(test_data, batch_size=batch_size)
        

    def count_test_dataset_num_class(self, test_data):
        y_all = []
        for _, y in test_data:
            y_all.append(y)
        y_count = collections.Counter(y_all)
        self.num_per_class = y_count.most_common() #[(label_id, num), (9, 1000), (2, 1000)... (5,1000)] NOT in oder
        self.num_class     = len(self.num_per_class) # num class

    def add_class_label(self, labels):
        self.labels = labels