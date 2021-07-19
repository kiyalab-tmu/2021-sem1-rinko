#download text data
import urllib.request
import nltk
nltk.download('punkt')

url = 'https://www.gutenberg.org/files/35/35-0.txt'
urllib.request.urlretrieve(url, 'text.txt')

#------------------------------------------------------------------------
#make word dictionary
f = open("text.txt", "r", encoding="utf_8")
f.read(1)
dic_w = {}
while True:
  line = f.readline()
  if line:
    tokenize = nltk.word_tokenize(line)
    for word in tokenize:
      if word not in dic_w:
        dic_w[word] = 1
      else: #既に辞書に存在する
        dic_w[word] = dic_w[word] + 1
  else:
    break
f.close()

#頻出順に並び替え
dic_w = sorted(dic_w.items(), key=lambda x:x[1], reverse=True)

#頻出順でインデックスを0から振る
idx = 0
dic_word = {}
for item in dic_w:
  k, v = item
  dic_word[k] = idx
  idx += 1

print(dic_word)
print(len(dic_word))


#make character dictionary
f = open("text.txt", "r", encoding="utf_8")
f.read(1)
dic = {}
while True:
  line = f.readline()
  if line:
    words = list(line)
    for word in words:
      if word not in dic:
        dic[word] = 1
      else:
        dic[word] = dic[word] + 1
  else:
    break

f.close()

#頻出順に並び替え
dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)

#インデックス0~の辞書を作成
idx = 0
dic_ch = {}
for item in dic:
  k, v = item
  dic_ch[k] = idx
  idx += 1

print(dic_ch)
print(len(dic_ch))
n_letters = len(dic_ch)


#--------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim

#keyからvalを求める関数
def key_from_val(val, dic=dic_ch):
  for k, v in dic.items():
    if v == val:
      return k

#word or char -> idx
def ch2idx(sentents, dic=dic_ch):
  dic = dic
  output = []
  sentents = list(sentents)
  for char in sentents:
    output.append(dic_ch[char])
  return output

#idx -> word or char
def idx2ch(idx_list, dic=dic_ch):
  dic = dic
  output = []
  idx_list = list(idx_list)
  for idx in idx_list:
    output.append(key_from_val(idx, dic=dic))
  return output


#--------------------------------------------------------------------
#model
class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    
    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(input_size + hidden_size, output_size)
    self.o2o = nn.Linear(hidden_size + output_size, output_size)
    self.dropout = nn.Dropout(0.1)
    self.softmax = nn.LogSoftmax(dim=1)

    #self.rnn = nn.RNN(input_size, hidden_size)
    #self.fc = nn.Linear(hidden_size, input_size)
  
  def forward(self, input, hidden):
    input_combined = torch.cat((input, hidden), 1)
    hidden = self.i2h(input_combined)
    output = self.i2o(input_combined)
    output_combined = torch.cat((hidden, output), 1)
    output = self.o2o(output_combined)
    output = self.dropout(output)
    output = self.softmax(output)
    return output, hidden

  def initHidden(self):
    return torch.zeros(1, self.hidden_size)
  
  
#-------------------------------------------------------------------
#create dataset
f = open("text.txt", "r", encoding="utf_8")
f.read(1)

dataset = []

batch_size = 1
sequence_len = 50
while True:
  line = f.read(sequence_len + 1)
  if line and len(line)==sequence_len+1:
    input = torch.zeros(sequence_len, 1, n_letters) #[30, 1, 92]

    line = list(line)
    line = ch2idx(line)

    #one-hot-vector
    for i in range(sequence_len):
      input[i][0][line[i]] = 1
    output = torch.tensor(line[1:])
    output.unsqueeze_(-1)
    dataset.append([input, output])

  else:
    break

print(len(dataset))
N = len(dataset)


#-------------------------------------------------------------------
#ここからmain
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_size = n_letters
hidden_size = 64
net = RNN(input_size, hidden_size, input_size)
net = net.to(device)

criterion = nn.NLLLoss()
lr = 0.0005

epochs = 30
n = 0

losses = []
perplexities = []

#学習パート
for epoch in range(epochs):
  print(f'epoch {epoch}')
  epoch_loss = 0
  epoch_perplexity = 0

  sentents = sample(idx2ch([random.randint(0, n_letters-1)])[0])
  for i in sentents:
    print(i, end='')

  for x, y in dataset:
    n += 1 #count

    x = x.to(device)
    y = y.to(device)

    hidden = net.initHidden()
    hidden = hidden.to(device)

    net.zero_grad()

    loss = 0
    perplexity = 0
    for i in range(x.size(0)):
      output, hidden = net(x[i], hidden)
      l = criterion(output, y[i])
      loss += l
      perplexity += torch.exp(loss)

    loss.backward()

    epoch_loss += loss.item() / x.size(0)
    epoch_perplexity += perplexity.item()

    losses.append(epoch_loss)
    perplexities.append(epoch_perplexity)

    
    for p in net.parameters():
      p.data.add_(p.grad.data, alpha=-lr)

    #途中経過
    if n % 1000 == 0:
      running_loss = (loss.item() / x.size(0))
      print(f'loss: {running_loss:.6f}, perplexity: {epoch_perplexity:.6f}')
      
      

#文章生成
device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_length = 200
#net.cpu()

def sample(start_letter='A', max_length=200):
  with torch.no_grad():
    start_letter = start_letter
    input = ch2idx(start_letter)
    hidden = net.initHidden()
    hidden = hidden.to(device)

    x = torch.zeros(1, 1, n_letters).to(device)
    x[0][0][input] = 1

    output_name = [start_letter]

    for i in range(max_length):
      output, hidden = net(x[0], hidden)
      topv, topi = output.topk(1)
      topi = topi[0][0].item()
      if topi == dic_ch['.']:
        break
      else:
        letter = idx2ch([topi])
        output_name += letter
      input = ch2idx(letter)
      x = torch.zeros(1, 1, n_letters).to(device)
      x[0][0][input] = 1

    return output_name


def samples(start_letters='ABC'):
  for start_letter in start_letters:
    sentents = sample(start_letter)
    for i in sentents:
      print(i, end='')
    print()


#samples('Hello')
