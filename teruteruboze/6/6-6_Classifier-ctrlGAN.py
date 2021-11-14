import argparse
import json
import torch
import util.makeDir as makeDir
import util.makeNet as makeNet
import torchvision.models as models
from torch import nn
import util.makeFigure as makeFigure

def main(args):
    # load json file #######################################################################
    with open(args.configGAN) as f:
        configGAN = json.load(f)
    with open(args.configClassifier) as f:
        configCLS = json.load(f)

    # make directories ####################################################################
    makeDir.do(
        args.logDir, 
        args.logfileName, 
        args.csvDir, 
        args.checkpointDir, 
        args.figureDir, 
        args.defaultDir)
    
    # CUDA setup ###########################################################################
    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # models ###############################################################################
    # GAN models
    # Generator
    modelGen   = makeNet.DCGeneratorV2(configGAN['training']['nCH'], 
                                       configGAN['training']['nFEATURE'],
                                       configGAN['training']['nZ'],
                                       configGAN['training']['nGPU']).to(device)
    modelGen.load_state_dict(torch.load(args.generatorModelDir + 'modelGen.ckpt'))

    # classifier model
    # must be the same as pre-trained classifier
    classifier = models.resnet18(pretrained=False)
    if configCLS['training']['nCH'] == 1:
        # to 1ch for MNIST
        classifier.conv1 = nn.Conv2d(
            configCLS['training']['nCH'], 
            64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False)
    # revise num_class in outlayer
    classifier.fc = nn.Linear(classifier.fc.in_features, configCLS['dataset']['num_class'])
    classifier = classifier.to(device)
    classifier.load_state_dict(torch.load(args.classifierModelDir + 'model.ckpt'))

    # output one fake image ###############################################################
    z = torch.randn(1, 100, 1, 1, device=device).requires_grad_()
    with torch.no_grad():
        fake = modelGen(z)
    p_number = classifier(fake).squeeze(0)[args.outNumber]
    makeFigure.imshow_text(
        fake.squeeze(0), 
        args.figureDir, 
        default_path=args.defaultDir, 
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        xlabel=f'p({args.outNumber})={p_number.sigmoid():.2f}')

    # update z to output desire number ###################################################
    z = z.detach().clone().requires_grad_()
    lr=args.lrZ
    fake_history=[]
    p_number_history = []
    i_iter = 0
    while p_number.sigmoid() < args.threshold:
        classifier.zero_grad()
        fake = modelGen(z)
        p_number = classifier(fake).squeeze(0)[args.outNumber]
        p_number.sigmoid().backward()
        z.data = z + (z.grad*lr)
        fake_history.append(fake.squeeze(0).detach())
        p_number_history.append(p_number.sigmoid())
        i_iter += 1
        if i_iter % args.displayIter == 0:
            print(f'num_iter: {i_iter}, p({args.outNumber})={p_number.sigmoid():.3f}')
        if i_iter == 5000:
            break
    print(f'num_iter: {i_iter}, p({args.outNumber})={p_number.sigmoid():.3f}')
    makeFigure.imshow_text(
            fake.squeeze(0).detach(), 
            args.figureDir, 
            default_path=args.defaultDir, 
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            fig_size=[4,4],
            fname='loop_last',
            xlabel=f'p({args.outNumber})={p_number.sigmoid():.3f}')

    if args.num_img != -1:
        img_fname = 1
        if args.displayFisrtIterImg != -1:
            len_img, len_p = args.displayFisrtIterImg, args.displayFisrtIterImg
        else:
            len_img, len_p = len(fake_history), len(p_number_history)
        for i in range(args.num_img+1, 1, -1):
            makeFigure.imshow_text(
                fake_history[len_img//i], 
                args.figureDir, 
                default_path=args.defaultDir, 
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                fig_size=[4,4],
                fname='loop_'+str(img_fname),
                xlabel=f'p({args.outNumber})={p_number_history[len_p//i]:.3f}')
            img_fname += 1

    
    

if __name__ == '__main__':
    path_cls = "4-12_ResNet_MNIST-pretrain_Augmentation_6epoch"
    path_gan = "6-5_WGAN_MNIST_100epoch_lr"
    fname_json_gan = "6-5_WGAN.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--configClassifier", type=str, default="./results/"+path_cls+"/exports/4-12_ResNet.json", help="path to training config json of clasifier")
    parser.add_argument("--classifierModelDir", type=str, default="./results/"+path_cls+"/exports/checkpoint/", help="path to model.ckpt in clasifier")

    parser.add_argument("--configGAN", type=str, default="./results/"+path_gan+"/exports/"+fname_json_gan, help="path to training config json of GAN")
    parser.add_argument("--generatorModelDir", type=str, default="./results/"+path_gan+"/exports/checkpoint/", help="path to modelGen.ckpt in GAN")

    parser.add_argument("--defaultDir", type=str, default="./results/6-6_controllableGAN_MNISTv7/exports/", help="path to output files (Main)")
    parser.add_argument("--logDir", type=str, default="log/", help="path to output log file")
    parser.add_argument("--logfileName", type=str, default="output.log", help="name of output log file")
    parser.add_argument("--csvDir", type=str, default="CSV/", help="path to output CSVs")
    parser.add_argument("--checkpointDir", type=str, default="checkpoint/", help="path to output checkpoint")
    parser.add_argument("--figureDir", type=str, default="figures/", help="path to output figures")

    parser.add_argument("--outNumber", type=int, default=0, help="output number in MNIST")
    parser.add_argument("--lrZ", type=float, default=0.2, help="learning rate of z")
    parser.add_argument("--threshold", type=float, default=0.9, help="probability of classifier")
    parser.add_argument("--displayIter", type=int, default=1000, help="print confidence of classifier per iter n;")
    parser.add_argument("--displayFisrtIterImg", type=int, default=10, help="save imgs per iter n; set -1 to disable")
    parser.add_argument("--num_img", type=int, default=6, help="save imgs until first n iter; set -1 to disable")
    args = parser.parse_args()
    #print(args)

    main(args)