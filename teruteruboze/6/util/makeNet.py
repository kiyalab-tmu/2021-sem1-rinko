import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax
import numpy as np
from torch.autograd import Variable

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 2))
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, 28 * 28), 
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y, x

class Autoencoder2(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 3))
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, 28 * 28), 
            nn.Tanh())

    def forward(self, x, code=None):
        if x != None:
            x = self.encoder(x)
        if code != None:
            x = code
        y = self.decoder(x)
        return y, x

class VAE(nn.Module):
    def __init__(self, d=20):
        super(VAE, self).__init__()

        self.d = d

        self.encoder = nn.Sequential(
            nn.Linear(784, self.d ** 2),
            nn.ReLU(),
            nn.Linear(self.d ** 2, self.d * 2),
            nn.ReLU(),
            nn.Linear(self.d * 2, 4),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d ** 2),
            nn.ReLU(),
            nn.Linear(self.d ** 2, 784),
            nn.Sigmoid(),
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, 784)).view(-1, 2, 2)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar, z


class DCGenerator(nn.Module):
    def __init__(self, ngpu=1, ngf=64, nz=100, nc=3):
        super(DCGenerator, self).__init__()
        self.ngpu = ngpu
        self.nz   = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    def makenoise_like(self, device):
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=device)

class DCDiscriminator(nn.Module):
    def __init__(self, ngpu=1, ndf=64, nc=3):
        super(DCDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# ===============================================

class DCGenerator_(nn.Module):
    def __init__(self, ngpu=1, ngf=64, nz=100, nc=1):
        super(DCGenerator_, self).__init__()
        self.ngpu = ngpu
        self.nz   = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 1, 1, 2, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    def makenoise_like(self, device):
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=device)

class DCDiscriminator_(nn.Module):
    def __init__(self, ngpu=1, ndf=64, nc=1):
        super(DCDiscriminator_, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# ===============================================

# For CIFAR10 (64x64) images ====================

class DCGeneratorV2(nn.Module):
    def __init__(self, nc, ngf=64, nz=100, ngpu=1):
        super(DCGeneratorV2, self).__init__()
        self.nc   = nc
        self.ngf  = ngf
        self.nz   = nz
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    def makenoise_like(self, device, size=None):
        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        if size == None:
            size = self.ngf
        self.fixed_noise = torch.randn(size, self.nz, 1, 1, device=device)

class DCDiscriminatorV2(nn.Module):
    def __init__(self, nc, ndf=64, ngpu=1):
        super(DCDiscriminatorV2, self).__init__()
        self.nc   = nc
        self.ndf  = ndf
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class WassersteinDCDiscriminator(nn.Module):
    def __init__(self, nc, ndf=64, ngpu=1):
        super(WassersteinDCDiscriminator, self).__init__()
        self.nc   = nc
        self.ndf  = ndf
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False)
            # nn.Sigmoid() # sigmoidを取る
        )

    def forward(self, input):
        output = self.main(input)
        output = output.mean(0)
        return output.view(1)

# ===============================================

class Generator_GMDC(nn.Module):
    def __init__(self, nz, nc):
        super(Generator_GMDC, self).__init__()
        self.nz = nz
        self.nc = nc

        net = nn.Sequential(
            nn.Linear(nz, 128),
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Linear(128, nc),
        )
        self.net = net
        self.nc = nc
        self.nz = nz
    
    def forward(self, input):
        output = self.net(input)
        return output

    def makenoise_like(self, device, size=None):
        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        if size == None:
            size = 10000
        self.fixed_noise = Variable(Tensor(np.random.normal(0, 1, (size, self.nz)).astype(np.float32))).to(device)

class Discriminator_GMDC(nn.Module):
    def __init__(self, nc):
        super(Discriminator_GMDC, self).__init__()
        self.nc = nc
        
        net = nn.Sequential(
            nn.Linear(nc, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.net = net
        self.nc = nc

    def forward(self, input):
        output = self.net(input)
        return output

class WassersteinDiscriminator_GMDC(nn.Module):
    def __init__(self, nc):
        super(WassersteinDiscriminator_GMDC, self).__init__()
        self.nc = nc
        
        net = nn.Sequential(
            nn.Linear(nc, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1),
        )
        self.net = net
        self.nc = nc

    def forward(self, input):
        output = self.net(input)
        return output

# ===============================================

class ConditionalDCGenerator(nn.Module):
    def __init__(self, nc, ngf=64, nz=100, ncls=10, ngpu=1):
        super(ConditionalDCGenerator, self).__init__()
        self.nc   = nc
        self.ngf  = ngf
        self.nz   = nz
        self.ncls = ncls
        self.ngpu = ngpu
        self.fc1  = nn.Linear(self.nz + self.ncls, self.nz)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise):
        concat = self.fc1(noise).view(-1, self.nz, 1, 1)
        return self.main(concat)

    def make_label(self, class_labels, device):
        batch_size = class_labels.size(0)
        self.makenoise_like(device, batch_size)
        z = self.fixed_noise
        y = torch.zeros(batch_size, self.ncls).scatter_(1, class_labels.view(-1, 1), 1).to(device)
        return torch.cat([z, y], dim=1)

    def makenoise_like(self, device, size=None):
        # Create batch of latent vectors that we will use to visualize
        # the progression of the generator
        if size == None:
            size = self.ngf
        self.fixed_noise = torch.randn(size, self.nz, device=device)

class ConditionalDCDiscriminator(nn.Module):
    def __init__(self, nc, ndf=64, ncls=10, ngpu=1):
        super(ConditionalDCDiscriminator, self).__init__()
        self.nc   = nc
        self.ndf  = ndf
        self.ncls = ncls
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.ncls + self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def make_input(self, data, class_labels, device):
        batch_size = data.shape[0]
        width = data.shape[2]
        height = data.shape[3]

        y = torch.zeros(batch_size, self.ncls, width * height).to(device)
        for i, num in enumerate(class_labels):
            y.data[i][num].fill_(1)
        y = y.view(-1, self.ncls, width, height)

        return torch.cat([data, y], dim=1)

# ===============================================

class Generator_(nn.Module):
    def __init__(self, nz=100, in_feature=28*28):
        super(Generator_, self).__init__()
        self.nz = nz
    
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, in_feature, bias=False),
            nn.ReLU(True),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.main(input)

    def makenoise_like(self, device):
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=device)

class Discriminator_(nn.Module):
    def __init__(self, in_feature=28*28):
        super(Discriminator_, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 384, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(384, 128, bias=False),
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 1, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.main(input)