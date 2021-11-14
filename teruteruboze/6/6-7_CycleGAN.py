import json
import shutil
import logging
import random
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import itertools
import numpy as np
import util.makeDir as makeDir
import util.makeDataset as makeDataset
import util.makeNet as makeNet
import util.trainCycleGAN as train
import util.makeFigure as makeFigure

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size=50):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

if __name__ == '__main__':

    # folder creation #######################################################################
    # config file (json)
    json_fname = '6-7_CycleGAN.json'
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
    if config['training']['inCH'] == 3:
        transform = transforms.Compose([
                    transforms.Resize(config['training']['imgSize']),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
    trainset   = makeDataset.Summer2WinterDataset(config['path']['dataset'], transform=transform, mode='train')
    trainloder = DataLoader(trainset, batch_size=config['training']['BATCH_SIZE'], shuffle=True)
    testset   = makeDataset.Summer2WinterDataset(config['path']['dataset'], transform=transform, mode='test')
    testloder = DataLoader(testset, batch_size=1, shuffle=False)
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
        Num Ch:           {config['training']['inCH']}
        Device:           {device}
    ''')

    # model ################################################################################
    modelGenA2B = makeNet.CycleGenerator(config['training']['inCH'], config['training']['outCH']).to(device)
    modelGenA2B.apply(makeNet.weights_init)
    modelGenB2A = makeNet.CycleGenerator(config['training']['outCH'], config['training']['inCH']).to(device)
    modelGenB2A.apply(makeNet.weights_init)

    modelDisA   = makeNet.CycleDiscriminator(config['training']['inCH']).to(device)
    modelDisA.apply(makeNet.weights_init)
    modelDisB   = makeNet.CycleDiscriminator(config['training']['outCH']).to(device)
    modelDisB.apply(makeNet.weights_init)

    # optimizer ############################################################################
    optimizerGen = torch.optim.Adam(
        itertools.chain(modelGenA2B.parameters(), modelGenB2A.parameters()), 
        lr=config['training']['lrG'],
        betas=(config['training']['beta'], 0.999))
    optimizerDis = torch.optim.Adam(
        itertools.chain(modelDisA.parameters(), modelDisB.parameters()), 
        lr=config['training']['lrG'],
        betas=(config['training']['beta'], 0.999))

    # losses ###############################################################################
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # train ################################################################################
    img_list = []
    Gen_loss = []
    Dis_loss = []

    # pool images
    fake_A_pool = ImagePool(config['training']['poolSize'])
    fake_B_pool = ImagePool(config['training']['poolSize'])

    for epoch in range(config['training']['EPOCH']):
        print(f'Epoch {epoch + 1}\n-------------------------------')
        logger.info(f'Epoch {epoch + 1}\n-------------------------------')
        # Train
        LossG, LossD = train.do(
            logger, trainloder, 
            modelGenA2B, modelGenB2A, modelDisA, modelDisB, 
            optimizerGen, optimizerDis, 
            criterion_GAN, criterion_cycle, criterion_identity,
            fake_A_pool, fake_B_pool,
            device)
        Gen_loss.append(LossG)
        Dis_loss.append(LossD)

    torch.save(modelGenA2B.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'modelGenA2B.ckpt')
    torch.save(modelGenB2A.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'modelGenB2A.ckpt')
    torch.save(modelDisA.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'modelDisA.ckpt')
    torch.save(modelDisB.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'modelDisB.ckpt')
    logger.info('PyTorch Model State Successfully saved to ' + config['path']['default'] + config['path']['ckpt'] + 'modelXXX.ckpt')
    
    
    # save for analysis ####################################################################
    # save Figs
    makeFigure.Fig_Gen_Dis(Gen_loss, Dis_loss, 'loss', config['path']['fig'], config['path']['default'])
    makeFigure.CycleGAN_imshow(testloder, modelGenA2B, modelGenB2A, device, config['path']['fig'], config['path']['default'])
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