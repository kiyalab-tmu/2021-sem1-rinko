import json
import logging
import torch
from torch import nn
import torchvision.utils as vutils
import numpy as np
import util.makeDir as makeDir
import util.makeDataset as makeDataset
import util.makeNet as makeNet
import util.trainDCGAN as train
import util.makeFigure as makeFigure

if __name__ == '__main__':

    # folder creation #######################################################################
    # config file (json)
    with open('6-4_DCGAN.json') as f:
        config = json.load(f)
    # make necessary dir
    makeDir.do(config['path']['log'], config['path']['log_Fname'],
               config['path']['csv'], config['path']['ckpt'], 
               config['path']['fig'], config['path']['default'])

    # CUDA setup ###########################################################################
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Dataset ##############################################################################
    dataset = makeDataset.BasicDataset(config['training']['BATCH_SIZE'],config['path']['dataset'])

    # logger setup #########################################################################
    logging.basicConfig(filename=config['path']['default']+config['path']['log']+config['path']['log_Fname'], 
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(config['proj']['date'] + ' - ' + config['proj']['name'])
    logger.info('comments: ' + config['proj']['comments'])
    logger.info(f'''Training setup:
        Epochs:           {config['training']['EPOCH']} epochs
        Batch size:       {config['training']['BATCH_SIZE']}
        Learning rate:    {config['training']['lr']}
        Device:           {device}
        All dataset:      {dataset.num_train}
        Train dataset:    {len(dataset.train)*int(config['training']['BATCH_SIZE'])}
        Valid dataset:    {dataset.num_valid}
    ''')

    # main #################################################################################
    modelGen   = makeNet.DCGenerator_().to(device)
    modelGen.apply(makeNet.weights_init)
    modelGen.makenoise_like(device)

    modelDis   = makeNet.DCDiscriminator_().to(device)
    modelDis.apply(makeNet.weights_init)

    loss_fn   = nn.BCELoss()
    optimizerGen = torch.optim.Adam(modelGen.parameters(), lr=config['training']['lr'], betas=(config['training']['beta'], 0.999))
    optimizerDis = torch.optim.Adam(modelDis.parameters(), lr=config['training']['lr'], betas=(config['training']['beta'], 0.999))

    img_list = []
    Gen_loss = []
    Dis_loss = []

    for epoch in range(config['training']['EPOCH']):
        print(f'Epoch {epoch + 1}\n-------------------------------')
        logger.info(f'Epoch {epoch + 1}\n-------------------------------')
        # Train
        LossG, LossD = train.do(logger, dataset.train, modelGen, modelDis, optimizerGen, optimizerDis, loss_fn, device)
        Gen_loss.append(LossG)
        Dis_loss.append(LossD)
        # 
        with torch.no_grad():
            fake = modelGen(modelGen.fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    torch.save(modelGen.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'modelGen.ckpt')
    torch.save(modelDis.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'modelDis.ckpt')
    logger.info('PyTorch Model State Successfully saved to ' + config['path']['default'] + config['path']['ckpt'] + 'modelXXX.ckpt')
    
    
    # save for analysis ####################################################################
    # save Figs
    makeFigure.Fig_Gen_Dis(Gen_loss, Dis_loss, 'loss', config['path']['fig'], config['path']['default'])
    makeFigure.saveAnimatedGAN2gif(img_list, config['path']['fig'], config['path']['default'])
    makeFigure.compareREALvsFAKE(dataset.train, img_list, device, config['path']['fig'], config['path']['default'])
    # save CSV
    Gen_loss = np.array(Gen_loss)
    np.savetxt(config['path']['default']+config['path']['csv']+'Gen_loss.csv', Gen_loss, delimiter=',')
    Dis_loss = np.array(Dis_loss)
    np.savetxt(config['path']['default']+config['path']['csv']+'Dis_loss.csv', Dis_loss, delimiter=',')

    # for log output #######################################################################
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)