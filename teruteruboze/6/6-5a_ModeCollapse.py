import json
import shutil
import logging
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import util.makeDir as makeDir
import util.makeDataset as makeDataset
import util.makeNet as makeNet
import util.trainWDCGAN_GMDC as train
import util.makeFigure as makeFigure

if __name__ == '__main__':

    # folder creation #######################################################################
    # config file (json)
    json_fname = '6-5a_ModeCollapse.json'
    with open(json_fname) as f:
        config = json.load(f)
    # make necessary dir
    makeDir.do(config['path']['log'], config['path']['log_Fname'],
               config['path']['csv'], config['path']['ckpt'], 
               config['path']['fig'], config['path']['default'])
    # copy json config file
    shutil.copy('./'+json_fname, config['path']['default']+json_fname)

    # CUDA setup ###########################################################################
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset ##############################################################################
    # データセットは TrainWDCGAN_GMDC.py で作ってます．
    # logger setup #########################################################################
    logging.basicConfig(filename=config['path']['default']+config['path']['log']+config['path']['log_Fname'], 
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(config['proj']['date'] + ' - ' + config['proj']['name'])
    logger.info('comments: ' + config['proj']['comments'])
    logger.info(f'''Training setup:
        Epochs:           {config['training']['EPOCH']} epochs
        Batch size:       {config['training']['BATCH_SIZE']}
        Learning rate(G): {config['training']['lrG']}
        Learning rate(D): {config['training']['lrD']}
        Num Ch:           {config['training']['nCH']}
        Device:           {device}
    ''')

    # main #################################################################################
    modelGen   = makeNet.Generator_GMDC(config['training']['nZ'], 
                                        config['training']['nCH']).to(device)
    modelGen.apply(makeNet.weights_init)

    modelDis   = makeNet.WassersteinDiscriminator_GMDC(config['training']['nCH']).to(device)
    modelDis.apply(makeNet.weights_init)

    optimizerGen = torch.optim.RMSprop(modelGen.parameters(), lr=config['training']['lrG'])
    optimizerDis = torch.optim.RMSprop(modelDis.parameters(), lr=config['training']['lrD'])

    Gen_loss = []
    Dis_loss = []

    for epoch in range(config['training']['EPOCH']):
        print(f'Epoch {epoch + 1}\n-------------------------------')
        logger.info(f'Epoch {epoch + 1}\n-------------------------------')
        # Train
        LossG, LossD = train.do(config, logger, modelGen, modelDis, optimizerGen, optimizerDis, device)
        Gen_loss.append(LossG)
        Dis_loss.append(LossD)

    torch.save(modelGen.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'modelGen.ckpt')
    torch.save(modelDis.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'modelDis.ckpt')
    logger.info('PyTorch Model State Successfully saved to ' + config['path']['default'] + config['path']['ckpt'] + 'modelXXX.ckpt')
    
    
    # save for analysis ####################################################################
    # save Figs
    modelGen.eval()
    with torch.no_grad():
            modelGen.makenoise_like(device)
            fake = modelGen(modelGen.fixed_noise).detach().cpu()
    real = train.gaussian_mixture_double_circle(10000,config['training']['NUM_CLUSTER'],config['training']['SCALE']).detach().cpu()
    makeFigure.Fig_Gen_Dis(Gen_loss, Dis_loss, 'loss', config['path']['fig'], config['path']['default'])
    makeFigure.plot_kde(real, config['path']['fig'], config['path']['default'], 'ked_groundtruth')
    makeFigure.plot_kde(fake, config['path']['fig'], config['path']['default'], 'ked_output')
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