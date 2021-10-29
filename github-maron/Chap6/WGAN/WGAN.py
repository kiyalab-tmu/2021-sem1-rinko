import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = MNIST("MNIST",train=True, download=True, transform=transform)

dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

workers = 2
nz = 100
nch_g = 128
nch_d = 128
n_epoch = 200
lr = 0.00005
clamp_lower = -0.01
clamp_upper = 0.01
n_critic = 5
outf = './WGAN/result-WGAN3'
display_interval = 600

G_losses = []
D_losses = []
D_x_out = []
D_G_z1_out = []

criterion = nn.BCELoss()

fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

class Generator(nn.Module):
    def __init__(self, nz=100, nch_g=128, nch=1):
        super().__init__()
        self.layers = nn.Sequential(
                nn.ConvTranspose2d(nz, nch_g * 4, 3, 1, 0),     
                nn.BatchNorm2d(nch_g * 4),                     
                nn.ReLU(),                          
                nn.ConvTranspose2d(nch_g * 4, nch_g * 2, 3, 2, 0),
                nn.BatchNorm2d(nch_g * 2),
                nn.ReLU(),
                nn.ConvTranspose2d(nch_g * 2, nch_g, 4, 2, 1),
                nn.BatchNorm2d(nch_g),
                nn.ReLU(),
                nn.ConvTranspose2d(nch_g, nch, 4, 2, 1),
                nn.Tanh()
        )


    def forward(self, z):
        return self.layers(z)
    
    
class Discriminator(nn.Module):
    def __init__(self, nch=1, nch_d=128):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(nch, nch_d, 4, 2, 1),   
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(nch_d, nch_d * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_d * 2),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(nch_d * 2, nch_d * 4, 3, 2, 0),
                nn.BatchNorm2d(nch_d * 4),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(nch_d * 4, 1, 3, 1, 0),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x.squeeze()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:          
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:    
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


netG = Generator(nz=nz, nch_g=nch_g).to(device)
netG.apply(weights_init) 
netD = Discriminator(nch_d=nch_d).to(device)
netD.apply(weights_init)

optimizerD = optim.RMSprop(netD.parameters(), lr=lr)  
optimizerG = optim.RMSprop(netG.parameters(), lr=lr)


for epoch in range(n_epoch):
    for itr, data in enumerate(dataloader):
        for n in range(n_critic):
            real_image = data[0].to(device)
            sample_size = real_image.size(0)
            
            
            noise = torch.randn(sample_size, nz, 1, 1, device=device)           
        
            netD.zero_grad() 
            output = netD(real_image)
            errD_real = torch.mean(output)
            D_x = output.mean().item()

            fake_image = netG(noise) 
            
            output = netD(fake_image.detach())
            errD_fake = torch.mean(output)
            D_G_z1 = output.mean().item()

            errD = -errD_real + errD_fake
            errD.backward() 
            optimizerD.step()

            for p in netD.parameters():
                p.data.clamp_(clamp_lower, clamp_upper)

            
        netG.zero_grad()
        noise = torch.randn(sample_size, nz, 1, 1, device=device)

        fake_image = netG(noise)
        output = netD(fake_image)
        errG = - torch.mean(output)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
    

        if itr % display_interval == 0: 
            print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(x): {:.3f} D(G(z)): {:.3f}/{:.3f}'
                  .format(epoch + 1, n_epoch,
                          itr + 1, len(dataloader),
                          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if epoch == 0 and itr == 0:  
            vutils.save_image(real_image.reshape(batch_size, 1, 28, 28), '{}/real_samples.png'.format(outf),
                              normalize=True, nrow=10)

       
        D_losses.append(errD.item())
        G_losses.append(errG.item())
        D_x_out.append(D_x)
        D_G_z1_out.append(D_G_z1)

  
    fake_image = netG(fixed_noise)  
    fake_image = fake_image.reshape(batch_size, 1, 28, 28)
    vutils.save_image(fake_image.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(outf, epoch + 1),
                      normalize=True, nrow=10)

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('%s.png' % ("WGAN_MNIST"))
