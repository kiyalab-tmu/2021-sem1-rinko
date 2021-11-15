import glob
import os
import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import itertools
from PIL import Image

from src.model import Discriminator, Generator, weights_init_normal

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

EPOCH_NUM = 5
batch_size = 1
dataroot = '../datasets/summer2winter_yosemite/'
lr = 0.0002
decay_epoch = 200
size = 256
input_nc = 3
output_nc = 3
device = f"cuda:{0}" if torch.cuda.is_available() else "cpu"

#Generator
netG_sum2win = Generator(input_nc, output_nc).to(device)
netG_win2sum = Generator(output_nc, input_nc).to(device)
# Discriminator
netD_sum = Discriminator(input_nc).to(device)
netD_win = Discriminator(output_nc).to(device)

#weight init
netG_win2sum.apply(weights_init_normal)
netG_sum2win.apply(weights_init_normal)
netD_win.apply(weights_init_normal)
netD_sum.apply(weights_init_normal)

#Loss
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

#optimizer
optimizer_G = optim.Adam(itertools.chain(netG_sum2win.parameters(), netG_win2sum.parameters()),
                                lr=lr, betas=(0.5, 0.999))
optimizer_D_win = optim.Adam(netD_win.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_sum = optim.Adam(netD_sum.parameters(), lr=lr, betas=(0.5, 0.999))

#data loader
transforms_ = [ transforms.Resize(int(size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
train_dl = DataLoader(ImageDataset(dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=batch_size, shuffle=True, num_workers=4)

for epoch in range(EPOCH_NUM):
    for i, datas in enumerate(train_dl):
        real_sum = datas['A'].to(device)
        real_win = datas['B'].to(device)
        size = datas['A'].shape[0]
        target_real = torch.ones(size, device=device, requires_grad=False)
        target_fake = torch.zeros(size, device=device, requires_grad=False)

        ##Generator
        optimizer_G.zero_grad()
        #Identity loss
        same_sum = netG_win2sum(real_sum)
        loss_identity_sum = criterion_identity(same_sum, real_sum)
        same_win = netG_sum2win(real_win)
        loss_identity_win = criterion_identity(same_win, real_win)
        #GAN loss
        fake_sum = netG_win2sum(real_win)
        pred_fake = netD_sum(fake_sum)
        loss_GAN_win2sum = criterion_GAN(pred_fake, target_real)
        fake_win = netG_sum2win(real_sum)
        pred_fake = netD_win(fake_win)
        loss_GAN_sum2win = criterion_GAN(pred_fake, target_real)
        #Consistency loss
        recovered_win = netG_sum2win(fake_sum)
        loss_cycle_winsumwin = criterion_cycle(recovered_win, real_win)*10
        recovered_sum = netG_win2sum(fake_win)
        loss_cycle_sumwinsum = criterion_cycle(recovered_sum, real_sum)*10

        loss_G = loss_identity_sum + loss_identity_win + loss_GAN_sum2win + loss_GAN_win2sum + loss_cycle_sumwinsum + loss_cycle_winsumwin
        loss_G.backward()

        optimizer_G.step()

        ##Discriminetor win
        optimizer_D_win.zero_grad()

        pred_real = netD_win(real_win)
        loss_D_real = criterion_GAN(pred_real, target_real)

        pred_fake = netD_win(fake_win.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        loss_D_win = (loss_D_real + loss_D_fake)*0.5
        loss_D_win.backward()

        optimizer_D_win.step()
        
        ##Discriminetor sum
        optimizer_D_sum.zero_grad()

        pred_real = netD_sum(real_sum)
        loss_D_real = criterion_GAN(pred_real, target_real)

        pred_fake = netD_sum(fake_sum.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        loss_D_sum = (loss_D_real + loss_D_fake)*0.5
        loss_D_sum.backward()

        optimizer_D_sum.step()

        if i % 20 == 0:
            print('Epoch[{}] loss_G: {:.4f} loss_G_identity: {:.4f} loss_G_GAN: {:.4f} loss_G_cycle: {:.4f} loss_D: {:.4f}'.format(
                epoch, loss_G, (loss_identity_win + loss_identity_sum),
                (loss_GAN_win2sum + loss_GAN_sum2win), (loss_cycle_winsumwin + loss_cycle_sumwinsum), (loss_D_win + loss_D_sum)
                ))
            torch.save(netG_win2sum.state_dict(), "checkpoints/CycleGAN_netG_win2sum.pth")
            torch.save(netG_sum2win.state_dict(), "checkpoints/CycleGAN_netG_sum2win.pth")
            torch.save(netD_win.state_dict(), "checkpoints/CycleGAN_netD_win.pth")
            torch.save(netD_sum.state_dict(), "checkpoints/CycleGAN_netD_sum.pth")
        

