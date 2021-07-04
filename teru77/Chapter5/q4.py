import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import q2 

partition=30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#define datasets
train_dataset = q2.TMDataset(partition=partition)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = q2.TMDataset(train=False,partition=partition)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

#define a model
class GRU(nn.Module):
    def __init__(self, n_input, n_hidden, n_vocab):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(n_vocab, n_hidden, padding_idx=0)
        self.drop1 = nn.Dropout(0.5)
        self.rnn = nn.GRU(n_hidden, n_hidden, num_layers=1, batch_first=True)
        self.out = nn.Linear(n_hidden, n_vocab)

    def forward(self, x, h=None):
        output = self.embedding(x)
        output = self.drop1(output)
        output, hp = self.rnn(output, h)
        output = self.drop1(output)
        output, hp = self.rnn(output, h)
        output = self.drop1(output)
        output = self.out(output)
        output = output[:, -1]
        return output
model = GRU(partition,256,37).to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100
train_losses = []
train_accuracies = []
train_perplexity = []
best_train_loss = None
best_train_acc = None

for epoch in range(epochs):  # loop over the dataset multiple times
    print("-"*30)
    print("epoch : {}".format(epoch))
    
    #train
    model.train()
    total_loss =0
    correct = 0
    total = 0
    total_perplexity = 0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [X, y]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        with torch.set_grad_enabled(True):
            pred_y = model(inputs.float())
            _, predicted = torch.max(pred_y, 1)
            loss=criterion(pred_y,labels)
            perplexity = torch.exp(loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_perplexity += perplexity.item()
        
        #acc
        correct += (predicted == labels.data).sum().item()  
        total += labels.size(0) 
        
    print(f"loss: {total_loss/len(train_dataloader)}")
    train_loss = total_loss/len(train_dataloader)
    train_losses.append(train_loss) #epochごとの平均lossを格納
    print(f"perplexity: {total_perplexity/len(train_dataloader)}")
    train_perplexity.append(total_perplexity/len(train_dataloader))
    train_acc = float(correct / total)
    print(f"acc: {train_acc}")
    train_accuracies.append(train_acc) #epochごとの平均accを格納
    
    if best_train_loss is None  or best_train_loss > train_loss:
        best_train_loss = train_loss
        best_train_acc =  train_acc

torch.save(model.state_dict(), './model.pth')
  
#test
model = GRU(partition,37)
model.load_state_dict(torch.load('./model.pth'))    
model = model.to(device)
model.eval()  

total_loss =0
correct = 0
total = 0
total_perplexity = 0
with torch.no_grad():
    for i,data in enumerate(test_dataloader,0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        pred_y = model(inputs.float())
        _, predicted = torch.max(pred_y, 1)
        #loss
        loss=criterion(pred_y,labels)
        total_loss += loss.item()
        #acc
        total += labels.size(0)
        correct += (predicted == labels.data).sum().item()
        #perplexity
        perplexity = torch.exp(loss)
        total_perplexity += perplexity.item()
    test_loss = total_loss/len(test_dataloader)
    test_acc = float(correct / total)
    test_perplexity = total_perplexity/len(test_dataloader)

# Plot result(loss)
plt.plot(range(1, epochs+1),train_losses,label="train")
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('./q4_loss.png')
plt.close()

# Plot result(acc)
plt.plot(range(1, epochs+1),train_accuracies,label="train")
plt.title('Accuracies')
plt.xlabel('epoch')
plt.ylabel('Acc')
plt.legend()
plt.savefig('./q4_acc.png')
plt.close()

# Plot result(acc)
plt.plot(range(1, epochs+1),train_perplexity,label="train")
plt.title('Perplexity')
plt.xlabel('epoch')
plt.ylabel('perplexity')
plt.legend()
plt.savefig('./q4_perplexity.png')
plt.close()

print("-"*30)
print(f"best_train_loss: {best_train_loss}")
print(f"best_train_acc: {best_train_acc}")
print(f"test_loss: {test_loss}")
print(f"test_acc: {test_acc}")
print(f"test_perplexity: {test_perplexity}")
