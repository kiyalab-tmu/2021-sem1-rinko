from torchvision.datasets import  mnist,cifar
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class DataLoad():

    def __init__(self):
        pass

    def load_data_mnist(self,batch_size=128):
        mnist_data = mnist.MNIST(root='./data',train=True,download=True,transform=transforms.Compose(
                                                 [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
        mnist_loader = DataLoader(mnist_data,batch_size=batch_size,shuffle=True)
        return mnist_loader

    def load_data_cifar10(self,batch_size=128):
        cifar_data = cifar.CIFAR10(root='./data',train=True,download=True,transform=transforms.Compose(
                                                 [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
        cifar_loader = DataLoader(cifar_data,batch_size=batch_size,shuffle=True)
        return cifar_loader