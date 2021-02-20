import sys

import config

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def printf(format, *args):
    sys.stdout.write(format % args)

def get_cifar10_dataloaders():
    assert config.DATA_PREPROCESSING == "IMAGENET" or config.DATA_PREPROCESSING == "CIFAR10", \
        "Data preprocessing type error, should be either 'IMAGENET' or 'CIFAR10'"

    if config.DATA_PREPROCESSING == "IMAGENET":
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]), download=True),
        batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)


        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])),
        batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)

    if config.DATA_PREPROCESSING == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ]), download=True),
            batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])),
            batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)

    return train_loader, val_loader

def get_cifar10_labels():
    return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
