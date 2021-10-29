import pickle
from matplotlib.animation import PillowWriter
import torch
import numpy as np
from torch.nn.modules import loss
def do(logger, dataset, modelGen, modelDis, optimizerGen, optimizerDis, loss_fn, device, real_label = 1., fake_label = 0.) :

    for i, (data, class_labels) in enumerate(dataset, 0):
        # train Discriminator ==========================
        # real
        modelDis.zero_grad()
        real = data.to(device)
        b_size = real.size(0)
        input_dis = modelDis.make_input(real, class_labels, device)
        output = modelDis(input_dis).view(-1)

        label = torch.full((b_size,), real_label, device=device)
        errDis_real = loss_fn(output, label)
        errDis_real.backward()
        D_x = output.mean().item()

        # fake 
        input_gen = modelGen.make_label(class_labels, device)
        fake = modelGen(input_gen)
        input_dis = modelDis.make_input(fake, class_labels, device)
        output = modelDis(input_dis.detach()).view(-1)

        label.fill_(fake_label)
        errDis_fake = loss_fn(output, label)
        errDis_fake.backward()
        D_G_z1 = output.mean().item()

        errDis = errDis_real + errDis_fake
        optimizerDis.step()

        # train Generater
        modelGen.zero_grad()
        output = modelDis(input_dis).view(-1)

        label.fill_(real_label)
        errGen = loss_fn(output, label)
        errGen.backward()
        D_G_z2 = output.mean().item()

        optimizerGen.step()

        if i % 50 == 0:
            logger.info('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (i, len(dataset), errDis.item(), errGen.item(), D_x, D_G_z1, D_G_z2))

    return errGen.item(), errDis.item()