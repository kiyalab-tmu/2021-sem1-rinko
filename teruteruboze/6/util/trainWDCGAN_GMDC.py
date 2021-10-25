import pickle
from matplotlib.animation import PillowWriter
from numpy.core.fromnumeric import size
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

def do(config, logger, modelGen, modelDis, optimizerGen, optimizerDis, device, real_label = 1., fake_label = 0.) :
    # この処理で，Generaterは1/5のデータしか見れないため，かなり不利に．
    i = 0
    while i < config['training']['NUM_ITER']:
        # train Discriminator ==============================================================================
        # reset requires_grad
        for p in modelDis.parameters():
            p.requires_grad = True # they are set to False below in netG update
        # DiscriminatorをGeneraterよりn回多く学習する
        j = 0
        while j < config['training']['nCritic'] and i < config['training']['NUM_ITER']:

            data = gaussian_mixture_double_circle(config['training']['BATCH_SIZE'],config['training']['NUM_CLUSTER'],config['training']['SCALE'])
            i += 1
            # real
            modelDis.zero_grad()
            real = data.to(device)
            errDis_real = modelDis(real/config['training']['SCALE'])

            # fake
            modelGen.makenoise_like(device=device, size=config['training']['BATCH_SIZE'])
            noise = modelGen.fixed_noise
            with torch.no_grad():
                fake = modelGen(noise)
            errDis_fake = modelDis(fake.detach()/config['training']['SCALE'])
            errDis = -torch.sum(errDis_real - errDis_fake) / config['training']['BATCH_SIZE']
            errDis.backward()
            optimizerDis.step()
            # clamp parameters to a cube
            for p in modelDis.parameters():
                p.data.clamp_(config['training']['discriminator_clamp'][0], config['training']['discriminator_clamp'][1])
            j += 1

        # tain Generater ================================================================================
        # reset requires_grad
        for p in modelDis.parameters():
                p.requires_grad = False # to avoid computation
        modelGen.makenoise_like(device=device, size=config['training']['BATCH_SIZE'])
        noise = modelGen.fixed_noise
        modelGen.zero_grad()
        fake = modelGen(noise)
        # errGen = -output; errGen.backward(mone)でも学習はOK．Loss表示の都合でこうしてる
        errGen = - torch.sum(modelDis(fake/config['training']['SCALE']) / config['training']['BATCH_SIZE'])
        errGen.backward()
        optimizerGen.step()

        if i % 50 == 0:
            logger.info('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (i, config['training']['NUM_ITER'], errDis.item(), errGen.item()))

    return errGen.item(), errDis.item()