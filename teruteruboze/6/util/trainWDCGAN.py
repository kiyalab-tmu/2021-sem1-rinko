import pickle
from matplotlib.animation import PillowWriter
from numpy.core.fromnumeric import size
import torch
import numpy as np
from torch.nn.modules import loss
def do(config, logger, dataset, modelGen, modelDis, optimizerGen, optimizerDis, device, real_label = 1., fake_label = 0.) :
    data_iter = iter(dataset) # load 1 batch
    # この処理で，Generaterは1/5のデータしか見れないため，かなり不利に．
    i = 0
    # The numbers 0 and -1
    one = torch.FloatTensor([1]).to(device)
    mone = one * -1
    while i < len(dataset):
        # train Discriminator ==============================================================================
        # reset requires_grad
        for p in modelDis.parameters():
            p.requires_grad = True # they are set to False below in netG update
        # DiscriminatorをGeneraterよりn回多く学習する
        j = 0
        while j < config['training']['nCritic'] and i < len(dataset):

            # clamp parameters to a cube
            for p in modelDis.parameters():
                p.data.clamp_(config['training']['discriminator_clamp'][0], config['training']['discriminator_clamp'][1])

            data = data_iter.next()
            i += 1
            # real
            modelDis.zero_grad()
            real = data[0].to(device)
            b_size = real.size(0)
            #label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = modelDis(real)

            errDis_real = output # BCEloss は使わない
            errDis_real.backward(one)

            # fake
            modelGen.makenoise_like(device=device, size=b_size)
            noise = modelGen.fixed_noise
            fake = modelGen(noise)
            #label.fill_(fake_label)
            output = modelDis(fake.detach())

            errDis_fake = output # BCEloss は使わない
            errDis_fake.backward(mone)
            optimizerDis.step()

            # errDis：print用とくに微分に使ったりはしない
            errDis = errDis_real - errDis_fake
            j += 1

        # tain Generater ================================================================================
        # reset requires_grad
        for p in modelDis.parameters():
                p.requires_grad = False # to avoid computation
        modelGen.zero_grad()
        #label.fill_(real_label)
        output = modelDis(fake)
        # errGen = -output; errGen.backward(mone)でも学習はOK．Loss表示の都合でこうしてる
        errGen = output
        errGen.backward(one)
        optimizerGen.step()

        if i % 50 == 0:
            logger.info('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (i, len(dataset), errDis.item(), errGen.item()))

    return errGen.item(), errDis.item()