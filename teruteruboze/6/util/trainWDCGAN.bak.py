import pickle
from matplotlib.animation import PillowWriter
import torch
import numpy as np
from torch.nn.modules import loss
def do(config, logger, dataset, modelGen, modelDis, optimizerGen, optimizerDis, device, real_label = 1., fake_label = 0.) :
    data_iter = iter(dataset) # load 1 batch
    # この処理で，Generaterは1/5のデータしか見れないため，かなり不利に．
    i = 0

    while i < len(dataset):
        # train Discriminator ==============================================================================
        # reset requires_grad
        # DiscriminatorをGeneraterよりn回多く学習する
        j = 0
        while j < config['training']['nCritic'] and i < len(dataset):
                
            data = data_iter.next()
            i += 1
            # real
            modelDis.zero_grad()
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = modelDis(real)

            errDis_real = output # BCEloss は使わない
            errDis_real.backward()
            D_x = output.mean().item()

            # fake 
            noise = torch.randn(b_size, modelGen.nz, 1, 1, device=device)
            with torch.no_grad():
                fake = modelGen(noise)
            label.fill_(fake_label)
            output = modelDis(fake.detach())

            errDis_fake = output # BCEloss は使わない
            errDis_fake.backward()
            D_G_z1 = output.mean().item()

            errDis = -errDis_real + errDis_fake
            optimizerDis.step()
            j += 1

            # clamp parameters to a cube
            for p in modelDis.parameters():
                p.data.clamp_(config['training']['discriminator_clamp'][0], config['training']['discriminator_clamp'][1])

        # tain Generater ================================================================================
        # reset requires_grad
        modelGen.zero_grad()
        label.fill_(real_label)
        output = modelDis(fake)

        errGen = -output # BCEloss は使わない
        errGen.backward()
        D_G_z2 = output.mean().item()

        optimizerGen.step()

        if i % 50 == 0:
            logger.info('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (i, len(dataset), errDis.item(), errGen.item(), D_x, D_G_z1, D_G_z2))

    return errGen.item(), errDis.item()