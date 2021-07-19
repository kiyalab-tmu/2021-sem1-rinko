import json
import logging
import torch
from torch import nn
import torchvision.models as models
import numpy as np
import util.makeDir as makeDir
import util.makeDataset as makeDataset
import util.train as train
import util.valid as valid
import util.test as test
import util.makeFigure as makeFigure

if __name__ == '__main__':

    # folder creation #######################################################################
    # config file (json)
    with open('4-10_GoogleNet.json') as f:
        config = json.load(f)
    # make necessary dir
    makeDir.do(config['path']['log'], config['path']['log_Fname'],
               config['path']['csv'], config['path']['ckpt'], 
               config['path']['fig'], config['path']['default'])

    # CUDA setup ###########################################################################
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Dataset ##############################################################################
    dataset = makeDataset.BasicDatasetTo3ch(config['training']['BATCH_SIZE'],config['path']['dataset'],config['dataset']['num_valid'])
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
        Train dataset:    {dataset.num_train}
        Valid dataset:    {dataset.num_valid}
    ''')

    # main #################################################################################
    model   = models.googlenet()
    model   = model.to(device)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'])

    train_loss = []
    train_acc  = []

    valid_loss = []
    valid_acc  = []

    prev_valid_loss = np.inf
    early_stopping  = 0

    for epoch in range(config['training']['EPOCH']):
        print(f'Epoch {epoch + 1}\n-------------------------------')
        logger.info(f'Epoch {epoch + 1}\n-------------------------------')
        # Train
        loss, acc = train.do(logger, dataset.train, model, loss_fn, optimizer, device)
        train_loss.append(loss.data.cpu().numpy())
        train_acc.append(acc)
        # Validation
        loss, acc = valid.do(logger, dataset.valid, model, loss_fn, optimizer, device)
        # Early stopping module
        if prev_valid_loss < loss.data.cpu().numpy():
            early_stopping += 1
            if early_stopping == config['training']['early_stopping_num']:
                logger.info(f'training suspended due to early stopping at Epoch {epoch}')
                print(f'Epoch {epoch}: training suspended due to early stopping.')
                logger.info(f' Epoch {epoch}: Early stopping was incremented')
                break
        else:
            prev_valid_loss = loss.data.cpu().numpy()
            early_stopping  = 0
            torch.save(model.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'model.ckpt')
            logger.info('PyTorch Model State Successfully saved to ' + config['path']['default'] + config['path']['ckpt'] + 'model.ckpt')
        valid_loss.append(loss.data.cpu().numpy())
        valid_acc.append(acc)

    
    # get y_pred and y_true for confusion matrix
    model.load_state_dict(torch.load(config['path']['default'] + config['path']['ckpt'] + 'model.ckpt')) # load the best model
    y_pred, y_true = test.do(logger, dataset.test, model, loss_fn, device)

    # save for analysis ####################################################################
    # save Figs
    makeFigure.Fig_confusion_matrix(y_pred, y_true, dataset.labels, config['path']['fig'], config['path']['default'])
    makeFigure.Fig_train_valid(train_loss, valid_loss, 'loss', config['path']['fig'], config['path']['default'])
    makeFigure.Fig_train_valid(train_acc,  valid_acc,   'acc', config['path']['fig'], config['path']['default'])
    # save CSV
    train_loss = np.array(train_loss)
    train_acc  = np.array(train_acc)
    np.savetxt(config['path']['default']+config['path']['csv']+'train_loss.csv', train_loss, delimiter=',')
    np.savetxt(config['path']['default']+config['path']['csv']+'train_acc.csv',  train_acc,  delimiter=',')
    valid_loss = np.array(valid_loss)
    valid_acc  = np.array(valid_acc)
    np.savetxt(config['path']['default']+config['path']['csv']+'valid_loss.csv', valid_loss, delimiter=',')
    np.savetxt(config['path']['default']+config['path']['csv']+'valid_acc.csv',  valid_acc,  delimiter=',')

    # for log output #######################################################################
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)