import pickle
from matplotlib.animation import PillowWriter
import torch
import numpy as np
import math

def gaussian_mixture_double_circle(batchsize, num_cluster=8, scale=2, std=0.2):
        rand_indices = np.random.randint(0, num_cluster, size=batchsize)
        base_angle = math.pi * 2 / num_cluster
        angle = rand_indices * base_angle - math.pi / 2
        mean = np.zeros((batchsize, 2), dtype=np.float32)
        mean[:, 0] = np.cos(angle) * scale
        mean[:, 1] = np.sin(angle) * scale
        # Doubles the scale in case of even number
        even_indices = np.argwhere(rand_indices % 2 == 0)
        mean[even_indices] /= 2
        return torch.Tensor(np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32))

def do(config, logger, modelGen, modelDis, optimizerGen, optimizerDis, loss_fn, device, real_label = 1., fake_label = 0.) :

    for i in range(config['training']['NUM_ITER']):
        # train Discriminator ==========================
        # real
        modelDis.zero_grad()
        data = gaussian_mixture_double_circle(config['training']['BATCH_SIZE'],config['training']['NUM_CLUSTER'],config['training']['SCALE'])
        real = data.to(device)
        b_size = config['training']['BATCH_SIZE']
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = modelDis(real).view(-1)

        errDis_real = loss_fn(output, label)
        errDis_real.backward()
        D_x = output.mean().item()

        # fake 
        modelGen.makenoise_like(device=device, size=b_size)
        noise = modelGen.fixed_noise
        fake = modelGen(noise)
        label.fill_(fake_label)
        output = modelDis(fake.detach()).view(-1)

        errDis_fake = loss_fn(output, label)
        errDis_fake.backward()
        D_G_z1 = output.mean().item()

        errDis = errDis_real + errDis_fake
        optimizerDis.step()

        # tain Generater
        modelGen.zero_grad()
        label.fill_(real_label)
        output = modelDis(fake).view(-1)

        errGen = loss_fn(output, label)
        errGen.backward()
        D_G_z2 = output.mean().item()

        optimizerGen.step()

        if i % 50 == 0:
            logger.info('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (i, config['training']['NUM_ITER'], errDis.item(), errGen.item(), D_x, D_G_z1, D_G_z2))

    return errGen.item(), errDis.item()