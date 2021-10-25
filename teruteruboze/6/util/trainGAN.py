import torch
import numpy as np
from torch.nn.modules import loss
def do(logger, dataset, modelGen, modelDis, optimizerGen, optimizerDis, loss_fn, device, real_label = 1., fake_label = 0.) :

    for i, data in enumerate(dataset, 0):
        # train Discriminator ==========================
        # real
        modelDis.zero_grad()
        real = data[0].to(device)
        b_size = real.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = modelDis(real).view(-1)

        errDis_real = loss_fn(output, label)
        errDis_real.backward()
        D_x = output.mean().item()

        # fake 
        noise = torch.randn(b_size, modelGen.nz, 1, 1, device=device)
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
                  % (i, len(dataset), errDis.item(), errGen.item(), D_x, D_G_z1, D_G_z2))

    return errGen.item(), errDis.item()