import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from PIL import Image

import glob as gF

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
        self.data = self.load_imgs()
    
    def load_imgs(self, path="../../data/summer2winter/"):
        img_list = []
        glob_path = gF.glob(path + 'fakeA/' +'*.jpg')
        for file_path in sorted(glob_path):
            img = Image.open(file_path)
            img_list.append(img)
        
        glob_path = gF.glob(path + 'fakeB/' +'*.jpg')
        for file_path in sorted(glob_path):
            img = Image.open(file_path)
            img_list.append(img)
        return img_list
    
    def __getitem__(self, index):
        out_data = self.data[index]
        if self.transform:
            out_data = self.transform(out_data)
        return out_data
    
    def __len__(self):
        return len(self.data)

class Summer2WinterDataset(torch.utils.data.Dataset):
    def __init__(self, img_path='../data/summer2winter/', transform=False, mode='train'):
        self.transform = transform
        self.domainA = self.load_imgs(img_path + mode + 'A/')
        self.domainB = self.load_imgs(img_path + mode + 'B/')
    
    def load_imgs(self, path):
        img_list = []
        glob_path = gF.glob(path+'*.jpg')
        for file_path in sorted(glob_path):
            img = Image.open(file_path)
            img_list.append(img)
        return img_list

    def __len__(self):
        return min(len(self.domainA), len(self.domainB))
    
    def __getitem__(self, idx):
        domainA = self.domainA[idx]
        domainB = self.domainB[idx]
        if self.transform:
            domainA = self.transform(domainA)
            domainB = self.transform(domainB)
        return {'A': domainA, 'B': domainB}


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    # cifar = dset.CIFAR10(root='data/', download=True,
    #                          transform=transforms.Compose([
    #                              transforms.Scale(32),
    #                              transforms.ToTensor(),
    #                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                          ])
    # )

    # IgnoreLabelDataset(cifar)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    summer2winter = Mydataset(transform=transform)

    print ("Calculating Inception Score...")
    print (inception_score(summer2winter, cuda=True, batch_size=32, resize=True, splits=10))
