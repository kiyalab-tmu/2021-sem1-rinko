import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#define dataset
w = np.array([2,-3.4])
b = 4.2
N = 1000
X = np.random.normal(0,1,(N,2))
noise = np.random.normal(0,0.01,N)
y = np.dot(X , w) + b + noise
y = y.reshape((-1,1))

# Plot features
plt.plot(X[0],X[1])
plt.savefig('./q1_feature.png')
plt.close()

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__() #data.Datasetを継承
        self.x = torch.from_numpy(X).clone().float()
        self.y = torch.from_numpy(y).clone().float()
        self.datanum = len(self.x)
    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]     

dataset =  Dataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=5,shuffle=True)     
    
#define model
class Net(nn.Module):
    def __init__(self,input_features,output_features):
        super().__init__() #nn.Moduleを継承
        self.linear = nn.Linear(input_features,output_features)
        nn.init.normal_(self.linear.weight, mean=0,std=0.01) #重みを初期化
    def forward(self,input):
        output = self.linear(input)
        return output
    
model = Net(2,1)

#define loss function
def squared_loss_function(pred_y,y):
    return torch.sum((y.reshape((pred_y.shape)) - pred_y)**2)/ 2

#define stochastic gradient descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 20
losses = []
for epoch in range(epochs):  # loop over the dataset multiple times
    print("epoch : {}".format(epoch))
    model.train()
    total_loss =0
    for i, data in enumerate(dataloader, 0):
        
        # get the inputs; data is a list of [X, y]
        X, y = data      
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred_y = model(X)
        loss=squared_loss_function(pred_y,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    losses.append(total_loss/i)#epochごとの平均lossを格納

#print learnt parameters
print("-----true parameters-----")
print(f"w：{w}")  
print(f"b：{b}")
print("-----learnt parameters-----")
print("w：",model.state_dict()["linear.weight"].detach().numpy().copy()[0])  #tensor -> numpy
print("b：",model.state_dict()["linear.bias"].detach().numpy().copy()[0])   #tensor -> numpy

# Plot result
plt.plot(range(1, epochs+1),losses)
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('./q1_loss.png')
plt.close()