# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 21:10:36 2021

@author: dciso
"""
import torch
from math import pi
from torch.utils.data import DataLoader
from model import Actor, Critic
from datasets import Composed, Fixed_Sines
from trainer import Trainer
import numpy as np
import os
import utils
import matplotlib.pyplot as plt


NUM_OF_CLASSES = 5

args = utils.load_args()

sines_args = args['sines']
arma_args = args['arma']

sines_args['frequency_range'][1] *= pi
sines_args['amplitude_range'][1] *= pi

dataset = Fixed_Sines(seed = sines_args['seed'])

if args['load_generator']:
    actor = Actor(args['num_classes'], args['size'], args['gen_embedding'])
    generator_path = args['generator_path']
    path = os.path.join('checkpoints', generator_path['timestamp'], generator_path['epoch'])
    g = utils.load_generator(path, actor)
    g = g.cuda()

else:
    actor = Actor(args['num_classes'], args['size'], args['gen_embedding'])
    actor_optim = torch.optim.RMSprop(actor.parameters(), lr=args['learning_rate'])

    critic = Critic(args['num_classes'], args['size'])
    critic_optim = torch.optim.RMSprop(critic.parameters(), lr=args['learning_rate'])

    train_labels_sines = torch.zeros(int(dataset.dataset.shape[0] / 2))
    train_labels = torch.tensor([])
    for i in range(NUM_OF_CLASSES):
        train_labels = torch.cat([train_labels, torch.tensor([i] * (int(sines_args['n_series']/NUM_OF_CLASSES)))])
    train_set = [
        (dataset.dataset[i], train_labels[i]) for i in range(len(dataset.dataset))
    ]
    dataloader = DataLoader(train_set, batch_size=args['batch_size'])
        # Instantiate Trainer
    trainer = Trainer(actor, critic, actor_optim, critic_optim, args['gp_weight'], args['critic_iterations'],
                             args['print_every'], torch.cuda.is_available(), args['checkpoint_frequency'])
        # Train model
    trainer_comp_3.train(dataloader, epochs=args['epochs'], plot_training_samples=True, checkpoint=args['checkpoint'])
    g = trainer.g

generated_data = utils.sample_generator(g,(5,50),torch.tensor([0,1,2,3,4]).long(),True)
generated_data = generated_data.cpu().detach().numpy()
data_real = dataset.dataset

utils.plot_results(data_real, generated_data)

data_real = [data_real[0], data_real[90], data_real[130], data_real[180]]
utils.gen_shapelets(data_real, [1,2,3,4], generated_data[1:], [1,2,3,4])
