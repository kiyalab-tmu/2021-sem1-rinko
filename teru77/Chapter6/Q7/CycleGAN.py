import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torchvision.utils as vutils
import itertools
import sys

from dataset import Dataset,data_transforms
from model import Generator,Discriminator

#os.environ["OMP_NUM_THREADS"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
sys.stdout.flush()

batch = 1
epoch_size = 200
#lr_d = 0.0002
#lr_g = 0.0002
lr = 0.0002
cycle_l = 10
identity_l =0.5
main_folder = "./result"

os.makedirs(main_folder, exist_ok=True)
os.makedirs(os.path.join(main_folder, "train"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "train/generated_images_A"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "train/generated_images_B"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "train/real_images_A"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "train/real_images_B"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test/generated_images_A"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test/generated_images_B"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test/real_images_A"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "test/real_images_B"), exist_ok=True)

save_path_train = os.path.join(main_folder, "loss_train.png")
save_path_test = os.path.join(main_folder, "loss_test.png")

traindataset = Dataset(root="/home/image/Cyclegan/dataset/apple2orange",mode="train",transform=data_transforms)
train_dataloader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=batch, shuffle=True)

testdataset = Dataset(root="/home/image/Cyclegan/dataset/apple2orange", mode="test",transform=data_transforms)
test_dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=batch, shuffle=True)

netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)

netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

optimizerG = torch.optim.Adam(itertools.chain(netG_A2B.parameters(),netG_B2A.parameters()),lr=lr,betas=(0.5, 0.999))
optimizerD_A = optim.Adam(netD_A.parameters(), lr = lr, betas=(0.5, 0.999))
optimizerD_B = optim.Adam(netD_B.parameters(), lr = lr, betas=(0.5, 0.999))

#criterion_GAN = nn.BCELoss()
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

loss_train_G_A2B = []
loss_train_G_B2A = []
loss_train_D_A = []
loss_train_D_B = []
loss_train_cycle = []
loss_train_identity = []

loss_test_G_A2B = []
loss_test_G_B2A = []
loss_test_D_A = []
loss_test_D_B = []
loss_test_cycle = []
loss_test_identity = []

def main(netG_A2B, netG_B2A, netD_A, netD_B, criterion_GAN, criterion_cycle, optimizerG, optimizerD_A, optimizerD_B, n_epoch, batch):

    for epoch in range(n_epoch):
        netG_A2B.train()
        netG_A2B.train()
        netD_A.train()
        netD_B.train()

        loss_train_G_A2B_epoch = 0
        loss_train_G_B2A_epoch = 0
        loss_train_D_A_epoch = 0
        loss_train_D_B_epoch = 0
        loss_train_cycle_epoch = 0
        loss_train_identity_epoch = 0

        # トレーニング
        for i, data_train in enumerate(train_dataloader, 0):
            if data_train[0].size()[0] != batch:
                break

            real_A = data_train[0].to(device)
            real_B = data_train[1].to(device)
            fake_A = netG_B2A(real_B)
            fake_B = netG_A2B(real_A)

            # Discriminator A
            optimizerD_A.zero_grad()
            # Real
            batch_size = real_A.size()[0]
            label = torch.ones(batch_size).to(device)
            output = netD_A(real_A)

            errD_A_real = criterion_GAN(output, label)
            errD_A_real.backward()

            # Fake
            label = torch.zeros(batch_size).to(device)
            output = netD_A(fake_A.detach())
            errD_A_fake = criterion_GAN(output, label)
            errD_A_fake.backward()

            loss_train_D_A_epoch += errD_A_real.item() + errD_A_fake.item()

            optimizerD_A.step()

            #Discriminator B
            optimizerD_B.zero_grad()

            # Real
            label = torch.ones(batch_size).to(device)
            output = netD_B(real_B)
            errD_B_real = criterion_GAN(output, label)
            errD_B_real.backward()

            # Fake
            label = torch.zeros(batch_size).to(device)
            output = netD_B(fake_B.detach())
            errD_B_fake = criterion_GAN(output, label)
            errD_B_fake.backward()

            loss_train_D_B_epoch += errD_B_real.item() + errD_B_fake.item()

            optimizerD_B.step()

            # Generator
            optimizerG.zero_grad()

            fake_A = netG_B2A(real_B)
            fake_B = netG_A2B(real_A)
            re_A = netG_B2A(fake_B)
            re_B = netG_A2B(fake_A)
            re_A2A = netG_B2A(real_A)
            re_B2B = netG_A2B(real_B)

            # GAN Loss
            label = torch.ones(batch_size).to(device)
            output1 = netD_A(fake_A)
            output2 = netD_B(fake_B)

            errG_B2A = criterion_GAN(output1, label)
            errG_A2B = criterion_GAN(output2, label)
            errG = errG_B2A + errG_A2B

            loss_train_G_B2A_epoch += errG_B2A.item()
            loss_train_G_A2B_epoch += errG_A2B.item()

            # cycle Loss
            loss_cycle = (criterion_cycle(re_A, real_A) + criterion_cycle(re_B, real_B)) * cycle_l
            loss_train_cycle_epoch += loss_cycle.item()
            #errG += loss_cycle

            # identity loss
            loss_identity = (criterion_identity(re_A2A, real_A) + criterion_identity(re_B2B, real_B)) * cycle_l * identity_l
            loss_train_identity_epoch += loss_identity.item()

            errG += loss_identity
            errG.backward()

            optimizerG.step()

        # epochの平均loss
        loss_D_A = loss_train_D_A_epoch/len(train_dataloader)
        loss_D_B = loss_train_D_B_epoch/len(train_dataloader)
        loss_G_A2B = loss_train_G_A2B_epoch/len(train_dataloader)
        loss_G_B2A = loss_train_G_B2A_epoch/len(train_dataloader)
        loss_cycle =loss_train_cycle_epoch/len(train_dataloader)
        loss_identity =loss_train_identity_epoch/len(train_dataloader)
        print("train epoch: [{0:d}/{1:d}] LossD_A: {2:.4f} LossD_B: {3:.4f} LossG_B2A: {4:.4f} LossG_A2B: {5:.4f} Loss_cycle: {6:.4f} Loss_identity: {7: 4f}".format(epoch+1, n_epoch, loss_D_A, loss_D_B, loss_G_B2A, loss_G_A2B, loss_cycle,loss_identity))
        sys.stdout.flush()

        joined_real_A = torchvision.utils.make_grid(real_A, nrow=2, padding=3)
        joined_real_B = torchvision.utils.make_grid(real_B, nrow=2, padding=3)

        joined_fake_A = torchvision.utils.make_grid(fake_A, nrow=2, padding=3)
        joined_fake_B = torchvision.utils.make_grid(fake_B, nrow=2, padding=3)

        vutils.save_image(joined_fake_A.detach(), os.path.join(main_folder, "train/generated_images_A/fake_samples_epoch_{0:03d}.png".format(epoch+1)),normalize=True)
        vutils.save_image(joined_fake_B.detach(), os.path.join(main_folder, "train/generated_images_B/fake_samples_epoch_{0:03d}.png".format(epoch+1)),normalize=True)
        vutils.save_image(joined_real_A, os.path.join(main_folder, "train/real_images_A/real_samples_epoch_{0:03d}.png".format(epoch+1)), normalize=True)
        vutils.save_image(joined_real_B, os.path.join(main_folder, "train/real_images_B/real_samples_epoch_{0:03d}.png".format(epoch+1)), normalize=True)

        loss_train_D_A.append(loss_D_A)
        loss_train_D_B.append(loss_D_B)
        loss_train_G_A2B.append(loss_G_A2B)
        loss_train_G_B2A.append(loss_G_B2A)
        loss_train_cycle.append(loss_cycle)
        loss_train_identity.append(loss_identity)

        # テスト
        with torch.no_grad():
            loss_test_G_A2B_epoch = 0
            loss_test_G_B2A_epoch = 0
            loss_test_D_A_epoch = 0
            loss_test_D_B_epoch = 0
            loss_test_cycle_epoch = 0
            loss_test_identity_epoch = 0
            for i, data_test in enumerate(test_dataloader, 0):
                real_A = data_test[0].to(device)
                real_B = data_test[1].to(device)
                fake_A = netG_B2A(real_B)
                fake_B = netG_A2B(real_A)

                #Discriminator A
                # Real
                batch_size = real_A.size()[0]
                label = torch.ones(batch_size).to(device)
                output = netD_A(real_A)
                errD_A_real = criterion_GAN(output, label)

                # Fake
                label = torch.zeros(batch_size).to(device)
                output = netD_A(fake_A.detach())
                errD_A_fake = criterion_GAN(output, label)

                loss_test_D_A_epoch += errD_A_real.item() + errD_A_fake.item()

                # Discriminator B
                # Real
                label = torch.ones(batch_size).to(device)
                output = netD_B(real_B)
                errD_B_real = criterion_GAN(output, label)

                # Fake
                label = torch.zeros(batch_size).to(device)
                output = netD_B(fake_B.detach())
                errD_B_fake = criterion_GAN(output, label)

                loss_test_D_B_epoch += errD_B_real.item() + errD_B_fake.item()

                # Generator
                fake_A = netG_B2A(real_B)
                fake_B = netG_A2B(real_A)
                re_A2A = netG_B2A(real_A)
                re_B2B = netG_A2B(real_B)
                re_A = netG_B2A(fake_B)
                re_B = netG_A2B(fake_A)


                # GAN Loss
                label = torch.ones(batch_size).to(device)
                output1 = netD_A(fake_A)
                output2 = netD_B(fake_B)

                errG_B2A = criterion_GAN(output1, label)
                errG_A2B = criterion_GAN(output2, label)

                loss_test_G_B2A_epoch += errG_B2A.item()
                loss_test_G_A2B_epoch += errG_A2B.item()

                # cycle Loss
                loss_cycle = (criterion_cycle(re_A, real_A) + criterion_cycle(re_B, real_B)) * cycle_l
                loss_test_cycle_epoch += loss_cycle.item()
                #errG += loss_cycle

                # identity loss
                loss_identity = (criterion_identity(re_A2A, real_A) + criterion_identity(re_B2B, real_B)) * cycle_l * identity_l
                loss_test_identity_epoch += loss_identity.item()

                errG += loss_identity

            # epochの平均loss
            loss_D_A = loss_test_D_A_epoch/len(test_dataloader)
            loss_D_B = loss_test_D_B_epoch/len(test_dataloader)
            loss_G_A2B = loss_test_G_A2B_epoch/len(test_dataloader)
            loss_G_B2A = loss_test_G_B2A_epoch/len(test_dataloader)
            loss_cycle =loss_test_cycle_epoch/len(test_dataloader)
            loss_identity =loss_test_identity_epoch/len(test_dataloader)
            print("test epoch: [{0:d}/{1:d}] LossD_A: {2:.4f} LossD_B: {3:.4f} LossG_B2A: {4:.4f} LossG_A2B: {5:.4f} Loss_cycle: {6:.4f} Loss_identity: {7: 4f}".format(epoch+1, n_epoch, loss_D_A, loss_D_B, loss_G_B2A, loss_G_A2B, loss_cycle,loss_identity))
            sys.stdout.flush()

            joined_real_A = torchvision.utils.make_grid(real_A, nrow=2, padding=3)
            joined_real_B = torchvision.utils.make_grid(real_B, nrow=2, padding=3)

            joined_fake_A = torchvision.utils.make_grid(fake_A, nrow=2, padding=3)
            joined_fake_B = torchvision.utils.make_grid(fake_B, nrow=2, padding=3)

            vutils.save_image(joined_fake_A.detach(), os.path.join(main_folder, "test/generated_images_A/fake_samples_epoch_{0:03d}.png".format(epoch+1)),normalize=True)
            vutils.save_image(joined_fake_B.detach(), os.path.join(main_folder, "test/generated_images_B/fake_samples_epoch_{0:03d}.png".format(epoch+1)),normalize=True)
            vutils.save_image(joined_real_A, os.path.join(main_folder, "test/real_images_A/real_samples_epoch_{0:03d}.png".format(epoch+1)), normalize=True)
            vutils.save_image(joined_real_B, os.path.join(main_folder, "test/real_images_B/real_samples_epoch_{0:03d}.png".format(epoch+1)), normalize=True)

            loss_test_D_A.append(loss_D_A)
            loss_test_D_B.append(loss_D_B)
            loss_test_G_A2B.append(loss_G_A2B)
            loss_test_G_B2A.append(loss_G_B2A)
            loss_test_cycle.append(loss_cycle)
            loss_test_identity.append(loss_identity)


    #保存
    torch.save(netG_A2B.state_dict(), os.path.join(main_folder, "netG_A2B.pth"))
    torch.save(netG_B2A.state_dict(), os.path.join(main_folder, "netG_B2A.pth"))
    torch.save(netD_A.state_dict(), os.path.join(main_folder, "netD_A.pth"))
    torch.save(netD_B.state_dict(), os.path.join(main_folder, "netD_B.pth"))

# プロット
def Plot_loss_train(save_path):
    fig = plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    x = [i for i in range(len(loss_train_G_A2B))]
    plt.plot(x, loss_train_G_A2B, color="r", label='G_B')
    plt.plot(x, loss_train_G_B2A, color="g", label='G_A')
    plt.plot(x, loss_train_D_A, color="b", label='D_A')
    plt.plot(x, loss_train_D_B, color="c", label='D_B')
    plt.plot(x, loss_train_cycle, color="y", label='cycle')
    plt.plot(x, loss_train_identity, color="m", label='identity')
    fig.legend()
    fig.savefig(save_path)


def Plot_loss_test(save_path):
    fig = plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    x = [i for i in range(len(loss_test_G_A2B))]
    plt.plot(x, loss_test_G_A2B, color="r", label='G_B')
    plt.plot(x, loss_test_G_B2A, color="g",label='G_A')
    plt.plot(x, loss_test_D_A, color="b", label='D_A')
    plt.plot(x, loss_test_D_B, color="c",label='D_B')
    plt.plot(x, loss_test_cycle, color="y", label='cycle')
    plt.plot(x, loss_test_identity, color="m", label='identity')
    fig.legend()
    fig.savefig(save_path)



if __name__ == '__main__':
    main(netG_A2B=netG_A2B, netG_B2A=netG_B2A, netD_A=netD_A, netD_B=netD_B, criterion_GAN=criterion_GAN, criterion_cycle=criterion_cycle, optimizerG=optimizerG, optimizerD_A=optimizerD_A, optimizerD_B=optimizerD_B, n_epoch=epoch_size, batch=batch)
    Plot_loss_train(save_path_train)
    Plot_loss_test(save_path_test)
    print("finish")