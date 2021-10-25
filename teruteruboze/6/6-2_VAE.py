import json
import logging
import torch
from torch import nn
import numpy as np
import util.makeDir as makeDir
import util.makeDataset as makeDataset
import util.makeNet as makeNet
import util.makeFunc as makeFunc
import util.trainVAE as trainVAE
import util.makeFigure as makeFigure


if __name__ == '__main__':

    # folder creation #######################################################################
    # config file (json)
    with open('5-2_VAE.json') as f:
        config = json.load(f)
    # make necessary dir
    makeDir.do(config['path']['log'], config['path']['log_Fname'],
               config['path']['csv'], config['path']['ckpt'], 
               config['path']['fig'], config['path']['default'])

    # CUDA setup ###########################################################################
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Dataset ##############################################################################
    dataset = makeDataset.BasicDatasetForVAE(config['training']['BATCH_SIZE'],config['path']['dataset'])
    dataset.add_class_label(config['dataset']['classes'])

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
    model   = makeNet.VAE()
    model   = model.to(device)

    loss_fn = makeFunc.VAE_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    train_loss = []

    prev_valid_loss = np.inf
    early_stopping  = 0

    for epoch in range(config['training']['EPOCH']):
        print(f'Epoch {epoch + 1}\n-------------------------------')
        logger.info(f'Epoch {epoch + 1}\n-------------------------------')
        # Train
        loss= trainVAE.do(logger, dataset.train, model, loss_fn, optimizer, device)
        train_loss.append(loss.data.cpu().numpy())

        if epoch + 1 == config['training']['init_model_save_epoch']:
            torch.save(model.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'init_model.ckpt')

    torch.save(model.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'model.ckpt')
    logger.info('PyTorch Model State Successfully saved to ' + config['path']['default'] + config['path']['ckpt'] + 'model.ckpt')
    
    # save for analysis ####################################################################
    # save Figs
    model.load_state_dict(torch.load(config['path']['default'] + config['path']['ckpt'] + 'model.ckpt')) # load the best model
    makeFigure.VAE2img(dataset.test, model, device, config['path']['fig'], config['path']['default'], 'final_epoch_img', 20)
    makeFigure.VAE2xy(dataset.test, model, device, config['path']['fig'], config['path']['default'], 'final_epoch_xy')
    makeFigure.Fig_train(train_loss, 'loss', config['path']['fig'], config['path']['default'])

    # save CSV
    train_loss = np.array(train_loss)
    np.savetxt(config['path']['default']+config['path']['csv']+'train_loss.csv', train_loss, delimiter=',')

    # for log output #######################################################################
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)