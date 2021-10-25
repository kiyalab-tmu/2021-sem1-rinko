import torch
import torch.nn as nn

class VAE_loss(nn.Module):
    def __init__(self):
        super(VAE_loss, self).__init__()

    def __call__(self, x_hat, x, mu, logvar, β=1):
        BCE = nn.functional.binary_cross_entropy(
            x_hat, x.view(-1, 784), reduction='sum'
        )
        KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

        return BCE + β * KLD
        
