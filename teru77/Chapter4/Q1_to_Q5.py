import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Q1(nn.Module):
    def __init__(self):
        super().__init__() #nn.Moduleを継承
        self.kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
        self.kernel = torch.as_tensor(self.kernel.reshape(1, 1, 3, 3))
    def forward(self, x):
        x = [F.conv2d(x[:, i:i + 1,:,:],self.kernel, padding=1) for i in range(3)]
        x = torch.cat(x, dim=1)#Q1
        return x
    
class Q2(nn.Module): #Q2
    def __init__(self):
        super().__init__() #nn.Moduleを継承
        self.edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
        self.edge_kernel = torch.as_tensor(self.edge_kernel.reshape(1, 1, 3, 3))
        self.gray_kernel = np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 3, 1, 1)
        self.gray_k = torch.as_tensor(self.gray_kernel)
    def forward(self, x):
        gray = torch.sum(x * self.gray_k, dim=1, keepdim=True)   
        edge_image = F.conv2d(gray, self.edge_kernel, padding=1)
        return edge_image
    
class Q3(nn.Module):
    def __init__(self):
        super().__init__() #nn.Moduleを継承
        self.kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
        self.kernel = torch.as_tensor(self.kernel.reshape(1, 1, 3, 3))
    def forward(self, x):
        x = [F.conv2d(x[:, i:i + 1,:,:],self.kernel, padding=10) for i in range(3)]
        x = torch.cat(x, dim=1)#Q3
        return x
    
class Q4(nn.Module):
    def __init__(self):
        super().__init__() #nn.Moduleを継承
        self.kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
        self.kernel = torch.as_tensor(self.kernel.reshape(1, 1, 3, 3))
    def forward(self, x):
        x = [F.conv2d(x[:, i:i + 1,:,:],self.kernel, stride=3) for i in range(3)]
        x = torch.cat(x, dim=1)#Q1
        return x
    
class Q5(nn.Module):
    def __init__(self):
        super().__init__() #nn.Moduleを継承
        self.Maxpool = nn.MaxPool2d(kernel_size = (8,8),stride=1) #Q5(maxプーリング層)
        self.Avgpool = nn.AvgPool2d(kernel_size = (8,8),stride=1) #Q5(averageプーリング層)    
    def forward(self, x):
        Max_pooling =  self.Maxpool(x)
        Avg_pooling =  self.Avgpool(x)
        return Max_pooling,Avg_pooling
 
def pooling(original_img,kernel_size,stride,padding,method="max"):
    img = original_img.copy()
    """
    Parameters:
        original_img: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        method: string, 'max' or 'avg'
    """
    # Padding
    img = np.pad(img, padding, mode='constant')

    output_shape = ((img.shape[0] - kernel_size)//stride + 1,
                    (img.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    img_w = as_strided(img,shape = output_shape + kernel_size, 
                        strides = (stride*img.strides[0],
                                   stride*img.strides[1]) + img.strides)
    img_w = img_w.reshape(-1, *kernel_size)
    
    if method== 'max':
        return img_w.max(axis=(1,2)).reshape(output_shape)
    elif method == 'avg':
        return img_w.mean(axis=(1,2)).reshape(output_shape)

if __name__ == '__main__':
    img = Image.open("./imori.jpg")
    array = np.asarray(img, np.float32).transpose([2, 0, 1]) / 255.0
    tensor = torch.as_tensor(np.expand_dims(array, axis=0)) 

    #Q1
    q1 = Q1() 
    output1 = q1(tensor)
    torchvision.utils.save_image(output1[0,:,:,:], "Q1.jpg")

    #Q2
    q2 = Q2() 
    output2 = q2(tensor)
    torchvision.utils.save_image(output2[0,:,:,:], "Q2.jpg")

    #Q3
    q3 = Q3() 
    output3 = q3(tensor)
    torchvision.utils.save_image(output3[0,:,:,:], "Q3.jpg")

    #Q4
    q4 = Q4() 
    output4 = q4(tensor)
    torchvision.utils.save_image(output4[0,:,:,:], "Q4.jpg")

    #Q5
    q5 = Q5() 
    Max_pooling,Avg_pooling = q5(tensor)
    torchvision.utils.save_image(Max_pooling[0,:,:,:], "Max_pooling.jpg")
    torchvision.utils.save_image(Avg_pooling[0,:,:,:], "Avg_pooling.jpg")

    #numpy
    A = np.random.randint(0,10,(4,4))
    print(A)
    max_pooling = pooling(A,2,1,0,method="max")
    avg_pooling = pooling(A,2,1,0,method="avg") 
    print(max_pooling)      
    print(avg_pooling) 