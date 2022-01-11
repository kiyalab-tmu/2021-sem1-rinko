import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image, ImageOps
import random
from matplotlib import cm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class Dataset(torch.utils.data.Dataset):
    classes = ["A", "B"]

    def __init__(self, root="/home/image/Cyclegan/dataset/apple2orange", mode="train",transform=None):
        self.transform = transform
        self.images_a = []
        self.images_b = []

        root_a_path = os.path.join(root, mode + "A")
        root_b_path = os.path.join(root, mode + "B")

        images_a0 = os.listdir(root_a_path)
        images_b0 = os.listdir(root_b_path)

        images_a0 = sorted(images_a0)
        images_b0 = sorted(images_b0)

        if len(images_a0)>len(images_b0):
            len_images = int(len(images_b0))
        else:
            len_images = int(len(images_a0))

        for i in range(len_images):
            self.images_a.append(os.path.join(root_a_path, images_a0[i]))
            self.images_b.append(os.path.join(root_b_path, images_b0[i]))

    def __getitem__(self, index):

        image_a_path = self.images_a[index]
        image_b_path = self.images_b[index]

        img_a = Image.open(image_a_path)
        img_b = Image.open(image_b_path)

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)


        return img_a, img_b

    def __len__(self):

        return len(self.images_a)