from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
import pylab
import seaborn as sns

def save(data, f_name, xlabel, ylabel, figs_path, default_path='./exports/'):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(list(range(len(data))), data, label=f_name)
    ax1.set_xlabel(xlabel)
    ax1.legend()
    ax1.set_ylabel(ylabel)
    ax1.set_title(ylabel + ' / epoch')
    fig1.savefig(default_path + figs_path + f_name  + '_graph.jpg')

def saveAnimatedGAN2gif(img_list, figs_path, default_path='./exports/'):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save(default_path + figs_path + 'AnimatedGAN.gif', writer="imagemagick")

def compareREALvsFAKE(dataloader, img_list, device,  figs_path, default_path='./exports/'):
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))

    plt.savefig(default_path + figs_path + 'REALvsFAKE.jpg')

def Fig_Gen_Dis(Gen, Dis, ylabel, figs_path, default_path='./exports/', xlabel='epoch'):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(list(range(len(Gen))), Gen, label='Generator ' + ylabel)
    ax1.plot(list(range(len(Dis))), Dis, label='Discriminator ' + ylabel)
    ax1.legend()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(ylabel + ' / epoch')
    fig1.savefig(default_path + figs_path + ylabel  + '_graph.jpg')

def Fig_train(train, ylabel, figs_path, default_path='./exports/', xlabel='epoch'):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.plot(list(range(len(train))), train, label='train ' + ylabel)
    ax1.legend()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(ylabel + ' / epoch')
    fig1.savefig(default_path + figs_path + ylabel  + '_graph.jpg')

# 関数名を confusion_matrix にすると、 cm = confusion_matrix() でsklearnの方ではなく、ここの関数を呼び出しやがる。気を付けて。
def Fig_confusion_matrix(y_pred, y_true, labels, figs_path, default_path='./exports/', f_name='confusion_matrix'):
    plt.figure(figsize=(12, 9))
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(float)
    cm /= np.sum(cm, axis=1)
    cm = pd.DataFrame(data=cm * 100, index=labels, columns=labels)
    seaborn.heatmap(cm, annot=True, cmap='Blues')
    plt.savefig(default_path + figs_path + f_name  + '_graph.jpg')

# AE用文字出力
def vec2img(x, num_img):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    # pick up num image
    x = x[0:num_img, :, :, :]
    x = x.view(num_img, 28, 28)
    return x

def AE2img(dataloader, model, device, figs_path, default_path='./exports/', fname='img', num_img=10):
    model.eval()
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.view(X.size(0), -1)
            X = X.to(device)
            out, _ = model(X)
            break

    # num_img分のミニバッチの画像を適当に抜き出す (注:num_img < BATCH_SIZE)
    # Test set の shuffle=true をおすすめ
    pic = vec2img(out.cpu().data, num_img)
    pic = pic.detach().numpy().copy()

    fig = plt.figure()
    for i in range(num_img):
        ipic = pic[i, :, :].reshape(28, 28)
        ax = fig.add_subplot(num_img//2, num_img//2,i+1)
        ax.imshow(ipic)
        plt.axis('off')

    plt.savefig(default_path + figs_path + fname)

def latent_vec2img(V, model, device, figs_path, default_path='./exports/', fname='latent_vec2img', num_img=1):
    model.eval()
    with torch.no_grad():
        V = V.view(V.size(0), -1)
        V = V.to(device)
        out, _ = model(None, V)

    # num_img分のミニバッチの画像を適当に抜き出す (注:num_img < BATCH_SIZE)
    # Test set の shuffle=true をおすすめ
    pic = vec2img(out.cpu().data, num_img)
    pic = pic.detach().numpy().copy()

    fig = plt.figure()
    for i in range(num_img):
        ipic = pic[i, :, :].reshape(28, 28)
        ax = fig.add_subplot(num_img//2, num_img//2,i+1)
        ax.imshow(ipic)
        plt.axis('off')

    plt.savefig(default_path + figs_path + fname)

def AE2xy(dataloader, model, device, figs_path, default_path='./exports/', fname='img', num_img=10):
    model.eval()
    isFirst = True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.view(X.size(0), -1)
            X, y = X.to(device), y.to(device)
            _, out = model(X)

            # to numpy
            out = out.cpu().detach().numpy().copy()
            y   = y.cpu().detach().numpy().copy()

            if isFirst:
                dataX = out
                dataY = y
                isFirst = False
            else:
                dataX = np.concatenate([dataX, out])
                dataY = np.append(dataY, y, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot()
    sc = ax.scatter(dataX[:,0], dataX[:,1], c=dataY, cmap='tab10')
    plt.colorbar(sc)
    #ax.legend(loc='upper left', borderaxespad=0, fontsize=18)
    plt.savefig(default_path + figs_path + fname + '1')


    fig = plt.figure()
    ax = fig.add_subplot()
    sc = ax.scatter(dataX[:,0], dataX[:,2], c=dataY, cmap='tab10')
    plt.colorbar(sc)
    plt.savefig(default_path + figs_path + fname + '2')

    fig = plt.figure()
    ax = fig.add_subplot()
    sc = ax.scatter(dataX[:,1], dataX[:,2], c=dataY, cmap='tab10')
    plt.colorbar(sc)
    plt.savefig(default_path + figs_path + fname + '3')

def VAE2img(dataloader, model, device, figs_path, default_path='./exports/', fname='img', num_img=10):
    model.eval()
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            out, _, _, _ = model(X)
            break

    # num_img分のミニバッチの画像を適当に抜き出す (注:num_img < BATCH_SIZE)
    # Test set の shuffle=true をおすすめ
    pic = vec2img(out.cpu().data, num_img)
    pic = pic.detach().numpy().copy()

    fig = plt.figure()
    for i in range(num_img):
        ipic = pic[i, :, :].reshape(28, 28)
        ax = fig.add_subplot(num_img//2, num_img//2,i+1)
        ax.imshow(ipic)
        plt.axis('off')

    plt.savefig(default_path + figs_path + fname)

def VAE2xy(dataloader, model, device, figs_path, default_path='./exports/', fname='img', num_img=10):
    model.eval()
    isFirst = True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.view(X.size(0), -1)
            X, y = X.to(device), y.to(device)
            _, _, _, out = model(X)

            # to numpy
            out = out.cpu().detach().numpy().copy()
            y   = y.cpu().detach().numpy().copy()

            if isFirst:
                dataX = out
                dataY = y
                isFirst = False
            else:
                dataX = np.concatenate([dataX, out])
                dataY = np.append(dataY, y, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot()
    sc = ax.scatter(dataX[:,0], dataX[:,1], c=dataY, cmap='tab10')
    plt.colorbar(sc)
    #ax.legend(loc='upper left', borderaxespad=0, fontsize=18)
    plt.savefig(default_path + figs_path + fname + '1')

def plot_kde(data, figs_path, default_path='./exports/', fname="kde", color="Greens"):
    fig = pylab.gcf()
    fig.set_size_inches(4.0, 4.0)
    pylab.clf()
    bg_color  = sns.color_palette(color, n_colors=256)[0]
    ax = sns.kdeplot(data[:, 0], data[:,1], shade=True, cmap=color, n_levels=30, clip=[[-4, 4]]*2)
    ax.set_facecolor(bg_color)
    kde = ax.get_figure()
    pylab.xlim(-4, 4)
    pylab.ylim(-4, 4)
    kde.savefig("{}/{}/{}.png".format(default_path, figs_path, fname))
    pylab.show()