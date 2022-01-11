import torch
from data_loader import DataLoad
from model import *
import torch.nn as nn
from torch.utils import tensorboard
from torch.autograd import Variable
from torchvision.utils import save_image,make_grid
from torchvision.models.inception import inception_v3
from torchvision import transforms
from torch.nn import functional as F
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'
k = 4
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128,type=int,help='Enter the batch size')
parser.add_argument('--total_epochs',default=500,type=int,help='Enter the total number of epochs')
parser.add_argument('--dataset',default='cifar10',help='Enter the dataset you want the model to train on')
parser.add_argument('--model_save_frequency',default=20,type=int,help='How often do you want to save the model state')
parser.add_argument('--image_sample_frequency',default=20,type=int,help='How often do you want to sample images ')
parser.add_argument('--learning_rate',default=0.0002,type=int)
parser.add_argument('--beta1',default=0.5,type=int,help='beta1 parameter for adam optimizer')
parser.add_argument('--beta2',default=0.999,type=int,help='beta2 parameter for adam optimizer')
parser.add_argument('--z_dim',default=100,type=int,help='Enter the dimension of the noise vector')
parser.add_argument('--exp_name',default='default-cifar10',help='Enter the name of the experiment')
args = parser.parse_args()

fixed_noise = torch.randn(50,args.z_dim,device=device)


#Create the experiment folder
if not os.path.exists(args.exp_name):
    os.makedirs(args.exp_name)

def load_data(use_data):
    # Initialize the data loader object
    data_loader = DataLoad()
    # Load training data into the dataloader
    if use_data == 'mnist':
        train_loader = data_loader.load_data_mnist(batch_size=args.batch_size)
    elif use_data == 'cifar10':
        train_loader = data_loader.load_data_cifar10(batch_size=args.batch_size)
    # Return the data loader for the training set
    return train_loader

def save_checkpoint(state,dirpath, epoch):
    #Save the model in the specified folder
    folder_path = dirpath+'/training_checkpoints'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = '{}-checkpoint-{}.ckpt'.format(args.dataset,epoch)
    checkpoint_path = os.path.join(folder_path, filename)
    torch.save(state, checkpoint_path)
    print(' checkpoint saved to {} '.format(checkpoint_path))

def generate_image(fakes,image_folder):
    #Function to generate image grid and save
    #image_grid = make_grid(fakes.to(device),padding=2,nrow=4,normalize=True)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    for i,image in enumerate(fakes):
        filename='{}/img_{}.png'.format(image_folder,i)
        img_pil = transforms.ToPILImage(mode='RGB')(image)
        img_pil.save(filename)

# Loss function
criterion = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(args.dataset)
discriminator = Discriminator(args.dataset)

if cuda:
    generator.cuda()
    discriminator.cuda()
    criterion.cuda()
    dtype = torch.cuda.FloatTensor
else :
    dtype = torch.FloatTensor

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1,args.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1,args.beta2))

# Establish convention for real and fake labels during training
real_label = float(1)
fake_label = float(0)

# Load training data
train_loader= load_data(args.dataset)

# Training Loop
# Lists to keep track of progress
# Create the runs directory if it does not exist
if not os.path.exists(args.exp_name+'/tensorboard_logs'):
    os.makedirs(args.exp_name+'/tensorboard_logs')
print("Starting Training Loop...")
sys.stdout.flush()

losses_D =[]
losses_G =[]
# For each epoch
for epoch in range(args.total_epochs):
    # Update the discriminator k times before updating generator as specified in the paper
    epoch_lossD = 0
    epoch_lossG = 0
    for i, (imgs, _) in enumerate(train_loader):
        # Format batch
        imgs = imgs.to(device)
        # Adversarial ground truths
        valid = Variable(torch.Tensor(imgs.size(0),1).fill_(real_label), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(imgs.size(0),1).fill_(fake_label), requires_grad=False).to(device)
        optimizer_D.zero_grad()
        # Calculate loss on all-real batch
        real_loss = criterion(discriminator(imgs), valid)
        # Generate batch of latent vectors
        noise = Variable(torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], args.z_dim)))).to(device)
        # Generate fake image batch with generator
        gen_imgs = generator(noise)
        # Classify all fake batch with D
        # Calculate D's loss on the all-fake batch
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        # Add the gradients from the all-real and all-fake batches
        loss_D = real_loss + fake_loss
        # Calculate the gradients
        loss_D.backward()
        #Update D
        optimizer_D.step()


        optimizer_G.zero_grad()
        # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        gen_imgs = generator(noise)
        output = discriminator(gen_imgs)
        # Calculate the probability of the discriminator to classify fake images as real.
        # If the  value of this probability is close to 0, then it means that the generator has
        # successfully learnt to fool the discriminator
        D_x = output.mean().item()
        # Calculate G's loss based on this output
        loss_G = criterion(output, valid)
        # Calculate gradients for G
        loss_G.backward()
        # Update G
        optimizer_G.step()
        # Output training stats
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\t'
                % (epoch+1, args.total_epochs, i+1, len(train_loader),
                    loss_D.item(), loss_G.item(), D_x))
        sys.stdout.flush()

        epoch_lossD += loss_D.item()
        epoch_lossG += loss_G.item()

    losses_D.append(epoch_lossD/i)
    losses_G.append(epoch_lossG/i)

    if (epoch+1) % args.model_save_frequency == 0:
    # Saved the model and optimizer states
        save_checkpoint({
            'epoch': epoch + 1,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_G' : optimizer_G.state_dict(),
            'optimizer_D' : optimizer_D.state_dict(),
        }, args.exp_name, epoch + 1)
    # Generate images from the generator network
    if (epoch+1) % args.image_sample_frequency == 0:
        with torch.no_grad():
            fakes = generator(fixed_noise)
            image_folder = args.exp_name + f'/genereated_images/{epoch}'
            generate_image(fakes,image_folder)

fig = plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Loss")
x = [i for i in range(len(losses_D))]
plt.plot(x, losses_D, color="r", label='D')
plt.plot(x, losses_G, color="g",label='G')
fig.legend()
fig.savefig("./loss.png")