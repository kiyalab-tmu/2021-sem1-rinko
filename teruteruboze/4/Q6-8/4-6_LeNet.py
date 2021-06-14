import json
import logging
import torch
from torch import nn
import numpy as np
import util.makeDir as makeDir
import util.makeDataset as makeDataset
import util.makeNet as makeNet
import util.train as train
import util.test as test
import util.makeFigure as makeFigure

if __name__ == '__main__':

    # folder creation #######################################################################
    # config file (json)
    with open('3-3_Multilayer_Perceptrons.json') as f:
        config = json.load(f)
    # make necessary dir
    makeDir.do(config['path']['log'], config['path']['log_Fname'],
               config['path']['csv'], config['path']['ckpt'], 
               config['path']['fig'], config['path']['default'])

    # logger setup #########################################################################
    logging.basicConfig(filename=config['path']['default']+config['path']['log']+config['path']['log_Fname'], 
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(config['proj']['date'] + ' - ' + config['proj']['name'])
    logger.info('comments: ' + config['proj']['comments'])
    logger.info(f'''Training setup:
        Epochs:           {config['training']['EPOCH']} epochs
        Batch size:       {config['training']['BATCH_SIZE']}
    ''')

    # main #################################################################################
    dataset = makeDataset.BasicDataset(config['training']['BATCH_SIZE'],config['path']['dataset'])
    dataset.add_class_label(config['dataset']['classes'])
    model   = makeNet.Multilayer_Perceptrons()
    model.weight_bias_init
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model   = model.to(device)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'])

    train_loss = []
    train_acc  = []

    for epoch in range(config['training']['EPOCH']):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        logger.info(f"Epoch {epoch + 1}\n-------------------------------")
        loss, acc = train.do(logger, dataset.train, model, loss_fn, optimizer, device)
        train_loss.append(loss.data.cpu().numpy())
        train_acc.append(acc)

    torch.save(model.state_dict(), config['path']['default'] + config['path']['ckpt'] + 'model.ckpt')
    print('PyTorch Model State Successfully saved to ' + config['path']['default'] + config['path']['ckpt'] + 'model.ckpt')
    # get y_pred and y_true for confusion matrix
    y_pred, y_true = test.do(logger, dataset.test, model, loss_fn, device)

    # save for analysis ####################################################################
    # save Figs
    makeFigure.Fig_confusion_matrix(y_pred, y_true, dataset.labels, config['path']['fig'], config['path']['default'])
    makeFigure.save(train_loss, 'train loss', 'epoch', 'loss', config['path']['fig'], config['path']['default'])
    makeFigure.save(train_acc, 'train acc', 'epoch', 'acc', config['path']['fig'], config['path']['default'])
    # save CSV
    train_loss = np.array(train_loss)
    train_acc  = np.array(train_acc)
    np.savetxt(config['path']['default']+config['path']['csv']+'train_loss.csv', train_loss, delimiter=',')
    np.savetxt(config['path']['default']+config['path']['csv']+'train_acc.csv',  train_acc,  delimiter=',')

    # for log output #######################################################################
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)