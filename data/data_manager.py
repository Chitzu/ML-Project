import json

import torchvision

from data.base_dataset import BaseDataset
import numpy as np
import torch
import torchvision.transforms as transforms


class DataManager:
    def __init__(self, config):
        self.config = config

    def get_dataloader(self, path):
        dataset = BaseDataset(path)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True if self.config['device'] == 'cuda' else False
        )
        return dataloader

    def get_train_eval_dataloaders(self):
        np.random.seed(707)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   batch_size=self.config['batch_size'],
                                                   pin_memory=True if self.config['device'] == 'cuda' else False)

        validation_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                        batch_size=self.config['batch_size'],
                                                        pin_memory=True if self.config['device'] == 'cuda' else False)
        return train_loader, validation_loader

    def get_train_eval_test_dataloaders(self):
        np.random.seed(707)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                               transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   batch_size=self.config['batch_size'],
                                                   pin_memory=True if self.config['device'] == 'cuda' else False)

        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                  batch_size=self.config['batch_size'],
                                                  pin_memory=True if self.config['device'] == 'cuda' else False)

        return train_loader, test_loader


