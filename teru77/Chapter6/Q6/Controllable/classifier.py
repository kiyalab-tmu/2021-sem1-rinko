from torchvision import transforms, datasets
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import itertools
import time
import numpy as np
from sklearn.metrics import f1_score
from models import NNet
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
sys.stdout.flush()

#変更したい特徴にインデックスをつける
all_attributes = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young'.split(' ')
relevant_attributes = ['Black_Hair', 'Blond_Hair']
relevant_indices = []
for i, a in enumerate(all_attributes):
    if a in relevant_attributes:
        relevant_indices.append(i)
relevant_indices = torch.tensor(relevant_indices)
index_to_string = {i:all_attributes[j.item()] for i,j in enumerate(relevant_indices)}

class Averager(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def gen_net_opt(lr, wd):
    classifier = NNet(len(relevant_attributes)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(),lr=lr,weight_decay=wd)
    return classifier, optimizer

def single_epoch(dataloader, model, loss_function,optimizer=None):

    if optimizer != None:
        model.train()
    else:
        model.eval()

    losses = Averager()
    f1 = [Averager() for _ in relevant_attributes]
    for xb, yb in tqdm(dataloader,leave=False):
        yb = yb[:,relevant_indices].float()
        y_hat = model(xb.to(device))
        y = yb.to(device)
        loss = loss_function(y_hat, y)

        if optimizer != None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        batch_size = len(yb)
        losses.update(loss.item(), batch_size)
        [p.update(f1_score(y[:,i].cpu(),(y_hat>0.5).int()[:,i].cpu()),batch_size) for i,p in enumerate(f1)]

    return losses.avg, [round(p.avg,2) for p in f1]

def fit(epochs, model, train_dl, valid_dl, loss_func, optimizer, scheduler=None):
    for epoch in range(epochs):
        print("="*30)
        print(f"epoch: {epoch}")
        start_time = time.time()
        train_loss, train_precisions = single_epoch(train_dl, model, loss_func, optimizer)
        if scheduler != None:
            scheduler.step()
        with torch.no_grad():
            valid_loss, valid_precisions = single_epoch(valid_dl, model, loss_func)
        secs = int(time.time() - start_time)
        print(f'Epoch {epoch} {secs}[sec]',end=' ')
        print(f'Train: loss {train_loss:.4f}. f1 {train_precisions}',end='\t')
        print(f'Valid: loss {valid_loss:.4f}. f1 {valid_precisions}')
        sys.stdout.flush()
    torch.save(classifier.state_dict(), 'save_classifier.pt')
    print('Save model as "save_classifier.pt"')

if __name__ == "__main__":
    mean,std=[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.CelebA(root="/home/image/CelebA/data",split='train',target_type='attr', download=False, transform=transform)
    valid_dataset = datasets.CelebA(root='/home/image/CelebA/data',split='valid',target_type='attr', download=False,transform=transform)

    train_dataloader = DataLoader(train_dataset,  batch_size=256,shuffle=True, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=256,shuffle=False, num_workers=1)
    classifier,optimizer = gen_net_opt(lr=0.00005,wd=0.005)

    fit(10,classifier,train_dataloader,valid_dataloader,nn.BCELoss(),optimizer)