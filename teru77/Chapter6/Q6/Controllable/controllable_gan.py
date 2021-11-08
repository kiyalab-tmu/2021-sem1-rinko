import torch
import matplotlib.pyplot as plt
from models import Generator,NNet
import sys

def controllable(mode: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    sys.stdout.flush()

    std,mean = (0.5, 0.5, 0.5),(0.5, 0.5, 0.5) # [0,1] -> [-1,1]

    # define generator
    generator = Generator(100).to(device)
    generator.load_state_dict(torch.load('save_generator.pt',map_location=torch.device(device)))
    generator.eval()

    # define classifier
    relevant_attributes = ['Black_Hair', 'Blond_Hair']
    classifier = NNet(len(relevant_attributes)).to(device)
    classifier.load_state_dict(torch.load('save_classifier.pt', map_location=torch.device(device)))
    classifier.eval()

    if mode == "black":
        INDEX = 0
    elif mode == "blond":
        INDEX = 1
    else:
        raise ValueError("The argument must be black or blond.")

    z = torch.randn(1, 100, 1, 1, device=device).requires_grad_()
    with torch.no_grad():
        fake = generator(z)
    hair_p = classifier(fake).squeeze(0)[INDEX]

    z = z.detach().clone().requires_grad_()
    lr=0.1
    fake_history=[]
    hair_p_history = []
    while hair_p.sigmoid() < 0.73:
        classifier.zero_grad()
        fake = generator(z)
        hair_p = classifier(fake).squeeze(0)[INDEX]
        hair_p.backward()
        z.data = z + (z.grad*lr)

        fake_history.append(fake.squeeze(0).detach())
        hair_p_history.append(hair_p.sigmoid())

        print(hair_p.sigmoid())
        sys.stdout.flush()
    skip = len(fake_history)//4
    fake_history = fake_history[0::skip]
    hair_p_history = hair_p_history[0::skip]
    fig, axes = plt.subplots(1, len(fake_history))
    for fake,p,ax in zip(fake_history,hair_p_history,axes):
        fake = fake.permute(1,2,0).cpu()
        fake = fake * torch.tensor(std) + torch.tensor(mean)
        ax.imshow(fake)
        ax.set_xlabel(f'p({mode})={p:.3f}',fontsize=28)
        ax.set_yticks([])
        ax.set_xticks([])
    fig.set_size_inches(50,50)
    plt.savefig(f"./{mode}.png")
    print("Save images")
    sys.stdout.flush()


if __name__ == '__main__':
    print("=========Generate black hair images=========")
    sys.stdout.flush()
    controllable(mode="black")

    print("=========Generate blond hair images=========")
    sys.stdout.flush()
    controllable(mode="blond")