import torch

def do(logger, dataset, modelGenA2B, modelGenB2A, modelDisA, modelDisB, optimizerGen, optimizerDis, criterion_GAN, criterion_cycle, criterion_identity, fake_A_pool, fake_B_pool, device, real_label = 1., fake_label = 0.):

    for i, data in enumerate(dataset, 0):
        # train Discriminator ==========================
        # real
        real_A = data['A'].to(device)
        real_B = data['B'].to(device)
        b_size = real_A.size(0)
        target_real = torch.full((b_size,1), real_label, dtype=torch.float, device=device)
        target_fake = torch.full((b_size,1), fake_label, dtype=torch.float, device=device)

        optimizerGen.zero_grad()

        # Identity loss
        # G_A2B(B)=B
        same_B = modelGenA2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A)=A
        same_A = modelGenB2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # Adversarial loss (fake)
        fake_B = modelGenA2B(real_A)
        pred   = modelDisB(fake_B)
        loss_GAN_A2B = criterion_GAN(pred, target_real)

        fake_A = modelGenB2A(real_B)
        pred = modelDisA(fake_A)
        loss_GAN_B2A = criterion_GAN(pred, target_real)

        # Cycle-consistency loss
        recovered_A = modelGenB2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = modelGenA2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # loss calculation
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        optimizerGen.step()

        # train Discriminator
        optimizerDis.zero_grad()
        # train Discriminator A
        # Real loss
        pred = modelDisA(real_A)
        loss_D_real = criterion_GAN(pred, target_real)
        # Fake loss
        fake_A = fake_A_pool.query(fake_A)
        pred = modelDisA(fake_A.detach())
        loss_D_fake = criterion_GAN(pred, target_fake)

        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        # train Discriminator B
        # Real loss
        pred_real = modelDisB(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        # Fake loss
        fake_B = fake_B_pool.query(fake_B)
        pred = modelDisB(fake_B.detach())
        loss_D_fake = criterion_GAN(pred, target_fake)

        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizerDis.step()

        if i % 50 == 0:
            logger.info('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_G_identity: %.4f\tLoss_G_GAN: %.4f\tloss_G_cycle:%.4f'
                  % (i, len(dataset),
                   (loss_D_A + loss_D_B), loss_G, 
                   (loss_identity_A + loss_identity_B), (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB)))

    return (loss_D_A.item() + loss_D_B.item()), loss_G.item()