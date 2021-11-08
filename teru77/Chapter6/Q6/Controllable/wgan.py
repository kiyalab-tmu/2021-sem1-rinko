from torchvision import transforms, datasets, utils
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm.notebook import tqdm
from torchvision.utils import save_image
from models import Generator,weights_init
import sys

#parameters
n_epochs=100
beta_1 = 0.5
beta_2 = 0.999
lr=0.0002
display_step = 500
z_dim = 100
c_lambda = 10
n_critic = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
sys.stdout.flush()

class Averager(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def critic_block(input_channels, output_channels, kernel_size, stride, padding, final_layer=False):
    if not final_layer:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, True))
    else:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding))

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            critic_block(3,128,4,2,1),
            critic_block(128,256,4,2,1),
            critic_block(256,512,4,2,1),
            critic_block(512,1024,4,2,1),
            critic_block(1024,1,4,1,0,final_layer=True))
        weights_init(self.layers)

    def forward(self, noise):
        return self.layers(noise)

def calc_gp(critic, real_images, fake_images):

    epsilon = torch.rand(len(real_images), 1, 1, 1, device=device, requires_grad=True)

    mixed_images = real_images * epsilon + fake_images * (1 - epsilon)
    mixed_scores = critic(mixed_images)

    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm-1)**2)

    return penalty


if __name__ == '__main__':
    mean,std=(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = datasets.CelebA(root="/home/image/CelebA/data",split='train',target_type='attr', transform=transform, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False,num_workers=4)

    generator = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))
    critic = Critic().to(device)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=(beta_1, beta_2))

    generator.train()
    critic.train()

    generator_avg_loss = Averager()
    critic_avg_loss = Averager()
    static_noise = torch.randn(32, z_dim, 1, 1, device=device)
    cur_step=0

    Loss_G_list = []
    Loss_C_list = []

    for epoch in range(n_epochs):
        for real_images, _ in tqdm(dataloader):
            real_images = real_images.to(device)
            current_bs = len(real_images)

            # Train critic
            critic.zero_grad()
            noise = torch.randn(current_bs, z_dim, 1, 1, device=device).float()
            fake_images = generator(noise)
            critic_fake_pred = critic(fake_images.detach()) # detach generator from graph
            critic_real_pred = critic(real_images)
            gp = calc_gp(critic, real_images, fake_images)
            critic_loss = torch.mean(critic_fake_pred) - torch.mean(critic_real_pred) + c_lambda*gp

            critic_avg_loss.update(critic_loss.item(), current_bs)

            critic_loss.backward()
            critic_opt.step()

            if cur_step % n_critic == 0:
                # Train generator
                gen_opt.zero_grad()
                noise = torch.randn(current_bs, z_dim, 1, 1, device=device).float()
                fake_images = generator(noise)
                critic_fake_pred = critic(fake_images)
                gen_loss = -torch.mean(critic_fake_pred)

                generator_avg_loss.update(gen_loss.item(), current_bs)

                gen_loss.backward()
                gen_opt.step()

            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Step {cur_step}: Generator loss: {generator_avg_loss.avg}, critic loss: {critic_avg_loss.avg}")
            cur_step += 1

        print(f"Epoch {epoch}: Generator loss: {generator_avg_loss.avg}, critic loss: {critic_avg_loss.avg}")
        sys.stdout.flush()
        Loss_G_list.append(generator_avg_loss.avg)
        Loss_C_list.append(critic_avg_loss.avg)

        with torch.no_grad():
            fake_images = generator(static_noise)
        save_image(utils.make_grid(fake_images,nrow=8), f'./wgan_img/celeba_wgan_epoch_{epoch}.jpg', normalize=True)

    torch.save(generator.state_dict(), 'save_generator.pt')

    plt.figure()
    plt.plot(range(len(Loss_G_list)), Loss_G_list, color='blue', linestyle='-', label='Generator Loss')
    plt.plot(range(len(Loss_C_list)), Loss_C_list, color='red', linestyle='-', label='Critic Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Generator Loss and Critic Loss')
    plt.grid()
    plt.savefig('Wgan_Loss_graph.png')