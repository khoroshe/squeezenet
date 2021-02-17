import sys

import config

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def printf(format, *args):
    sys.stdout.write(format % args)

def get_cifar10_dataloaders():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
    batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)

    return train_loader, val_loader
