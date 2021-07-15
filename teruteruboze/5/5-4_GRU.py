import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os
import logging

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, hidden_state.detach()

    
def train():
    # my custom variables
    record_loss = []
    record_perplexcity = []
    min_loss = np.inf
    early_stopping = 0
    ########### Hyperparameters ###########
    hidden_size = 256   # size of hidden state
    seq_len = 100       # length of LSTM sequence
    num_layers = 2      # number of layers in RNN
    lr = 0.002          # learning rate
    epochs = 100        # max number of epochs
    op_seq_len = 200    # total num of characters in output test sequence
    load_chk = False    # load weights from save_path directory to continue training
    early_stopping_num = 10
    save_path = "./time-machine_GRU_3/"
    data_path = "./data/time-machine-train.txt"
    #######################################

    ######### Directory Creation ##########
    try:
        os.makedirs(save_path)
    except FileExistsError:
        pass
    #######################################

    # logger setup ########################
    logging.basicConfig(filename=save_path+"log.txt", level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f'''Training setup:
        Epochs:           {epochs} epochs
        Batch size:       {1}
        Larning rate:     {lr}
        Hidden size:      {hidden_size}
    ''')
    
    # load the text file
    data = open(data_path, 'r').read()
    chars = sorted(list(set(data)))
    data_size, vocab_size = len(data), len(chars)
    print("----------------------------------------")
    print("Data has {} characters, {} unique".format(data_size, vocab_size))
    print("----------------------------------------")
    logger.info("----------------------------------------")
    logger.info("Data has {} characters, {} unique".format(data_size, vocab_size))
    logger.info("----------------------------------------")
    
    # char to index and index to char maps
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
    # convert data from chars to indices
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]
    
    # data tensor on device
    data = torch.tensor(data).to(device)
    data = torch.unsqueeze(data, dim=1)
    
    # model instance
    rnn = RNN(vocab_size, vocab_size, hidden_size, num_layers).to(device)
    
    # load checkpoint if True
    if load_chk:
        rnn.load_state_dict(torch.load(save_path))
        print("Model loaded successfully !!")
        print("----------------------------------------")
    
    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    
    # training loop
    for i_epoch in range(1, epochs+1):
        
        # random starting point (1st 100 chars) from data to begin
        data_ptr = np.random.randint(100)
        n = 0
        running_loss = 0
        runing_perplexcity = 0
        hidden_state = torch.zeros(2,1,hidden_size).to(device)
        
        while True:
            input_seq = data[data_ptr : data_ptr+seq_len]
            target_seq = data[data_ptr+1 : data_ptr+seq_len+1]
            
            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)
            
            # compute loss
            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
            running_loss += loss.item()
            runing_perplexcity += torch.exp(loss).item()
            
            # compute gradients and take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update the data pointer
            data_ptr += seq_len
            n +=1
            
            # if at end of data : break
            if data_ptr + seq_len + 1 > data_size:
                break
            
        # print loss and save weights after every epoch
        print("Epoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss/n))
        logger.info("Epoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss/n))
        # model save
        record_loss.append(running_loss/n)
        record_perplexcity.append(runing_perplexcity/n)
        if i_epoch == 1:
            torch.save(rnn.state_dict(), save_path+"model_first_epoch.ckpt")
        if min_loss > (running_loss/n):
            min_loss = running_loss/n
            torch.save(rnn.state_dict(), save_path+"model_best_epoch.ckpt")
        else:
            early_stopping += 1
            if early_stopping == early_stopping_num:
                print("Training suspended due to earliy stopping at epoch {}".format(i_epoch))
                logger.info("Training suspended due to earliy stopping at epoch {}".format(i_epoch))
                break
        
        
        # sample / generate a text sequence after every epoch
        data_ptr = 0
        hidden_state = torch.zeros(2,1,hidden_size).to(device)
        
        # random character from data to begin
        rand_index = np.random.randint(data_size-1)
        input_seq = data[rand_index : rand_index+1]
        
        print("----------------------------------------")
        logger.info("----------------------------------------")
        sentence = ''
        while True:
            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)
            
            # construct categorical distribution and sample a character
            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample()
            
            # the sampled character
            sentence += ix_to_char[index.item()] + ''
            
            # next input is current output
            input_seq[0][0] = index.item()
            data_ptr += 1
            
            if data_ptr > op_seq_len:
                break
        print(sentence)
        logger.info(sentence)
        print("\n----------------------------------------")
        logger.info("----------------------------------------")
    
    record_loss = np.array(record_loss)
    np.savetxt(save_path+'loss.csv',  record_loss,  delimiter=',')
    record_perplexcity = np.array(record_perplexcity)
    np.savetxt(save_path+'perplexcity.csv',  record_perplexcity,  delimiter=',')

    # for log output ######################
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
        
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train()


