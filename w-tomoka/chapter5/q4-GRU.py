import urllib.request
import nltk
nltk.download('punkt')

url = 'https://www.gutenberg.org/files/35/35-0.txt'
urllib.request.urlretrieve(url, 'text.txt')

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

dic_w = sorted(dic_w.items(), key=lambda x:x[1], reverse=True)

idx = 0
dic_word = {}
for item in dic_w:
  k, v = item
  dic_word[k] = idx
  idx += 1

print(dic_word)
print(len(dic_word))


#make character dictionary--------------------------------------------------

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

dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)

idx = 1 #idx=0は<EOS>
dic_ch = {}
dic_ch['<EOS>'] = 0
for item in dic:
  k, v = item
  dic_ch[k] = idx
  idx += 1

print(dic_ch)
print(len(dic_ch))
n_letters = len(dic_ch)


import torch
import torch.nn as nn
import torch.optim as optim

def key_from_val(val, dic=dic_ch):
  for k, v in dic.items():
    if v == val:
      return k

def ch2idx(sentents, dic=dic_ch):
  dic = dic
  output = []
  sentents = list(sentents)
  for char in sentents:
    output.append(dic_ch[char])
  return output

def idx2ch(idx_list, dic=dic_ch):
  dic = dic
  output = []
  idx_list = list(idx_list)
  for idx in idx_list:
    output.append(key_from_val(idx, dic=dic))
  return output

#model
class GRU(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, batch_size, batch_first=False):
    super(GRU, self).__init__()
    self.hidden_size = hidden_size

    self.gru = nn.GRU(input_size, hidden_size, batch_first=batch_first)
    self.fc = nn.Linear(hidden_size, output_size)
  
  def forward(self, input, hidden):
    output, h = self.gru(input, hidden)
    output = self.fc(output[:, -1])
    return output, h

  def initHidden(self):
    hidden = torch.zeros(1, 1, self.hidden_size)
    hidden = torch.zeros(1, 64, self.hidden_size)
    #hidden = torch.zeros(1, batch_size, self.hidden_size)
    #hidden = torch.tensor(hidden, dtype=torch.long, device=device)
    return hidden



def sample(start_letter='A', max_length=200):
  with torch.no_grad():
    start_letter = start_letter
    input = ch2idx(start_letter)
    #hidden = net.initHidden() #batch処理ではないのでこれは使わない
    hidden = torch.zeros(1, 1, 64)
    hidden = hidden.to(device)

    x = torch.zeros(1, 1, n_letters).to(device)
    x[0][0][input] = 1

    output_name = [start_letter]

    for i in range(max_length):
      output, hidden = net(x, hidden) #(1,1,92),(1,1,64) 
      topv, topi = output.topk(1)
      topi = topi[0][0].item()
      #if topi == dic_ch['.']:
      if topi == dic_ch['<EOS>']:
        letter = idx2ch([topi])
        output_name += letter
        break
      else:
        letter = idx2ch([topi])
        output_name += letter
      input = ch2idx(letter)
      x = torch.zeros(1, 1, n_letters).to(device)
      x[0][0][input] = 1

    return output_name


def samples(start_letters='ABC', max_length=200):
  for start_letter in start_letters:
    sentents = sample(start_letter, max_length)
    for i in sentents:
      print(i, end='')
    print()
    
    
#dataset----------------------------------------------------
f = open("text.txt", "r", encoding="utf_8")
f.read(1)

dataset = []

batch_size = 1
sequence_len = 50
while True:
  line = f.read(sequence_len)
  if line and len(line)==sequence_len:
    input = torch.zeros(sequence_len, batch_size, n_letters) #[30, 1, 92]

    line = list(line)
    line = ch2idx(line)

    for i in range(sequence_len):
      input[i][0][line[i]] = 1
    input = input.unsqueeze(0)
    #input.shape -> (1, 50, 1, 92)
    output = line[1:]
    output.append(0)
    output = torch.tensor(output)
    dataset.append([input, output])

  else:
    break

print(len(dataset))
N = len(dataset)
print(dataset[0][1])

#データセットをバッチごとにまとめる

import random

split = 0.7
i = 0
train_data_idx = []
val_data_idx = []
train_dataset = []
val_dataset = []
while i < int(N * split):
  n = random.randint(0, N-1)
  if not n in train_data_idx:
    train_dataset.append(dataset[n])
    train_data_idx.append(n)
    i += 1
print(len(train_dataset))
print(len(train_dataset[0]))
print(len(train_dataset[0][0]))
print(len(train_dataset[0][0][0]))
print(train_dataset[0][0][0])

for j in range(N-1):
  if not j in train_data_idx:
    val_dataset.append(dataset[j])
    val_data_idx.append(j)
    
    
#main--------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

input_size = n_letters
hidden_size = 64
net = GRU(input_size, hidden_size, input_size, batch_size=1)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

epochs = 100
n = 0

train_losses = []
train_perplexities = []
val_losses = []
val_perplexities = []

for epoch in range(epochs):
  print(f'epoch {epoch}, lr: {scheduler.get_last_lr()[0]:.6f}')

  #epochの初めに文章を生成
  sentents = sample(idx2ch([random.randint(1, n_letters-1)])[0])
  for i in sentents:
    print(i, end='')
  print()
  
  #train
  train_loss = 0
  train_perplexity = 0
  for x, y in train_dataset:
    x = x.to(device)
    y = y.to(device)
    
    hidden = net.initHidden().to(device)

    optimizer.zero_grad()
    output, hidden = net(x[0], hidden)
    loss = criterion(output, y)
    perplexity = torch.exp(loss / x.size(0)).item()

    loss.backward()
    optimizer.step()

    train_loss += loss.item() / x.size(0)
    train_perplexity += perplexity

  train_loss = train_loss / len(train_dataset)
  train_perplexity = train_perplexity / len(train_dataset)

  train_losses.append(train_loss)
  train_perplexities.append(train_perplexity)

  #test
  val_loss = 0
  val_perplexity = 0
  for x, y in val_dataset:
    x = x.to(device)
    y = y.to(device)
    hidden = net.initHidden().to(device)

    output, hidden = net(x[0], hidden)
    loss = criterion(output, y)
    perplexity = torch.exp(loss / x.size(0)).item()

    val_loss += loss.item() / x.size(0)
    val_perplexity += perplexity
  val_loss = val_loss / len(val_dataset)
  val_perplexity = val_perplexity / len(val_dataset)

  val_losses.append(val_loss)
  val_perplexities.append(val_perplexity)

  print(f'train_loss: {train_loss:.6f}, train_perplexity: {train_perplexity:.6f}',
        f'val_loss: {val_loss:.6f}, val_perplexity: {val_perplexity:.6f}')
  scheduler.step()

#plot
import matplotlib.pyplot as plt

plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='val loss')
plt.legend()
plt.show()

plt.plot(train_perplexities, label='train perplexity')
plt.plot(val_perplexities, label='val acc')
plt.legend()
plt.show()


#文章生成---------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

max_length = 100
#net.cpu()

for n in range(10):
  sentents = samples('Iabcde', max_length)
  for i in sentents:
    print(i, end='')
  print()
