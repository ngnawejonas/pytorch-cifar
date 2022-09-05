import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image

from typing import Tuple, Any

class MNIST_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        # self.transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.1307,), (0.3081,))])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        # if not isinstance(x, np.ndarray):
        #     x = Image.fromarray(x.numpy(), mode='L')
        #     x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y, train=True):
        self.X = X
        self.Y = Y
        if train:
            self.transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x)
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)