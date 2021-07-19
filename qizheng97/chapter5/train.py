import util
import rnn
import torch
from torch import nn
import random
import matplotlib.pyplot as plt
import numpy as np


lines = util.read()
print('sentences %d' % len(lines))

tokens = util.tokenize(lines)
print(tokens[7])
print(tokens[8])
vocab = util.Vocab(tokens)
print(len(vocab.token_to_idx.items()),list(vocab.token_to_idx.items())[1:10])

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = util.load_data()
print(vocab_size)
#RNN
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
gru_layer=nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
lstm_layer=nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = rnn.RNNModel(rnn_layer, vocab_size).to(device)
a=rnn.predict_rnn_pytorch('this', 10, model, vocab_size, device, idx_to_char, char_to_idx)
num_epochs, num_steps, batch_size, lr, clipping_theta = 500, 35, 32, 0.001, 1e-2
pred_period, pred_len, prefixes = 50, 1000, ['time tra', 'time','space']
loss_rnn,perplexity_rnn=rnn.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,corpus_indices, idx_to_char, char_to_idx,num_epochs, num_steps, lr, clipping_theta,batch_size, pred_period, pred_len, prefixes)

#GRU
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
gru_layer=nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
lstm_layer=nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = rnn.RNNModel(gru_layer, vocab_size).to(device)
a=rnn.predict_rnn_pytorch('this', 10, model, vocab_size, device, idx_to_char, char_to_idx)



loss_gru,perplexity_gru=rnn.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,corpus_indices, idx_to_char, char_to_idx,num_epochs, num_steps, lr, clipping_theta,batch_size, pred_period, pred_len, prefixes)

#LSTM
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
gru_layer=nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
lstm_layer=nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
model = rnn.RNNModel(lstm_layer, vocab_size).to(device)
a=rnn.predict_rnn_pytorch('this', 10, model, vocab_size, device, idx_to_char, char_to_idx)



loss_lstm,perplexity_lstm=rnn.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,corpus_indices, idx_to_char, char_to_idx,num_epochs, num_steps, lr, clipping_theta,batch_size, pred_period, pred_len, prefixes)

plt.figure(1)
plt.plot(np.arange(num_epochs), loss_rnn,label="RNN")
plt.plot(np.arange(num_epochs), loss_gru,label="GRU")
plt.plot(np.arange(num_epochs), loss_lstm,label="LSTM")
plt.legend(loc=0,ncol=1)
plt.xlabel("epoches")
plt.ylabel("loss")
plt.figure(2)
plt.plot(np.arange(num_epochs),perplexity_rnn,label="RNN")
plt.plot(np.arange(num_epochs),perplexity_gru,label="GRU")
plt.plot(np.arange(num_epochs),perplexity_lstm,label="LSTM")
plt.legend(loc=0,ncol=1)
plt.xlabel("epoches")
plt.ylabel("perplexity")
plt.show()