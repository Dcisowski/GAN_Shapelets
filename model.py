# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 00:14:00 2021

@author: dciso
"""
from torch import nn
import torch
from torch.nn.utils import spectral_norm
from utils import AddDimension, SqueezeDimension


def create_generator_architecture(embed_size):
    return nn.Sequential(nn.Linear(50+embed_size, 100),
                         nn.LeakyReLU(0.2, inplace=True),
                         AddDimension(),
                         spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
                         nn.Upsample(200),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Upsample(400),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Upsample(800),

                         spectral_norm(nn.Conv1d(32, 1, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),

                         SqueezeDimension(),
                         nn.Linear(800, 100)
                         )


def create_critic_architecture():
    return nn.Sequential(spectral_norm(nn.Conv1d(1+1, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.MaxPool1d(2),
                         
                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.MaxPool1d(2),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Flatten(),

                         nn.Linear(800, 50),
                         nn.LeakyReLU(0.2, inplace=True),

                         nn.Linear(50, 15),
                         nn.LeakyReLU(0.2, inplace=True),

                         nn.Linear(15, 1)
                         )


class Actor(nn.Module):
    def __init__(self, num_classes, _size, embed_size):
        super().__init__()
        self.main = create_generator_architecture(embed_size)
        self.embed = nn.Embedding(num_classes, embed_size)

    def forward(self, input, labels):
        embedding = self.embed(labels)
        input = torch.cat([input, embedding], dim=1)
        return self.main(input)


class Critic(nn.Module):
    def __init__(self, num_classes, _size):
        super().__init__()
        self.main = create_critic_architecture()
        self._size = _size
        self.embed = nn.Embedding(num_classes,_size)
    def forward(self, input, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self._size)
        input = input.unsqueeze(1)
        input = torch.cat([input, embedding], dim=1)
        return self.main(input)
