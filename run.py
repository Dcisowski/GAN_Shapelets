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
args2 = utils.load_args2()
    # Instantiate Generator and Critic + initialize weights
#g_1 = Actor(args['num_classes'],args['size'],args['gen_embedding'])

#g_opt_1 = torch.optim.RMSprop(g_1.parameters(), lr=args['learning_rate'])
#d_1 = Critic(args['num_classes'],args['size'])

#d_opt_1 = torch.optim.RMSprop(d_1.parameters(), lr=args['learning_rate'])

    # Create Dataloader

#sines_args = args['sines']
#arma_args = args['arma']

#sines_args['frequency_range'][1] *= pi
#sines_args['amplitude_range'][1] *= pi

#dataset = Composed(sines_args, arma_args)
#train_labels_sines = torch.zeros(int(dataset.dataset.shape[0] / 2))
#train_labels_arma = torch.ones(int(dataset.dataset.shape[0] / 2))
#train_labels = torch.cat((train_labels_sines,train_labels_arma))
#train_set = [
#    (dataset.dataset[i], train_labels[i]) for i in range(len(dataset.dataset))
#]
#print('Composed Dataset Chosen')
#dataloader = DataLoader(train_set, batch_size=args['batch_size'])

    # Instantiate Trainer
#trainer_comp_1 = Trainer(g_1, d_1, g_opt_1, d_opt_1,args['print_and_save'],args['gp_weight'], args['critic_iterations'],
#                  args['print_every'], torch.cuda.is_available(), args['checkpoint_frequency'])
    # Train model
#print('Training is about to start...')

#trainer_comp_1.train(dataloader, epochs=args['epochs'], plot_training_samples=True, checkpoint=args['checkpoint'])

#g_2 = Actor(args2['num_classes'],args2['size'],args2['gen_embedding'])

#g_opt_2 = torch.optim.RMSprop(g_2.parameters(), lr=args2['learning_rate'])
#d_2 = Critic(args2['num_classes'],args2['size'])

#d_opt_2 = torch.optim.RMSprop(d_2.parameters(), lr=args2['learning_rate'])

    # Create Dataloader

sines_args = args2['sines']
arma_args = args2['arma']

sines_args['frequency_range'][1] *= pi
sines_args['amplitude_range'][1] *= pi

dataset = Composed(sines_args, arma_args)
#train_labels_sines = torch.zeros(int(dataset.dataset.shape[0] / 2))
#train_labels_arma = torch.ones(int(dataset.dataset.shape[0] / 2))
#train_labels = torch.cat((train_labels_sines,train_labels_arma))
#train_set = [
#    (dataset.dataset[i], train_labels[i]) for i in range(len(dataset.dataset))
#]
#print('Composed Dataset Chosen')
#dataloader = DataLoader(train_set, batch_size=args2['batch_size'])

    # Instantiate Trainer
#trainer_comp_2 = Trainer(g_2, d_2, g_opt_2, d_opt_2,args2['gp_weight'], args2['critic_iterations'],
#                  args2['print_every'], torch.cuda.is_available(), args2['checkpoint_frequency'])
    # Train model

#trainer_comp_2.train(dataloader, epochs=args2['epochs'], plot_training_samples=True, checkpoint=args2['checkpoint'])
#path = os.path.join('checkpoints','_Aug_27_2021_01_34_45_', '22800_epoch.pkl')
#g = utils.load_generator(path,g_1)
#g = g.cuda()
#dat = utils.sample_generator(g,(8,50),torch.Tensor([0,0,0,0,1,1,1,1]).long(),True)
#dat_test = utils.sample_generator(g,(8,50),torch.Tensor([0,1,0,0,1,0,1,1]).long(),True)

#dataset = dataset.dataset
#real_dat = np.concatenate((dataset[0:4],dataset[250:254]),axis=0)
#real_dat_test = np.concatenate((dataset[8:12],dataset[260:264]),axis=0)
#generated_dat = dat.cpu().detach().numpy()
#generated_dat_test = dat_test.cpu().detach().numpy()
#labels_gen = [0,0,0,0,1,1,1,1]
#labels_gen_test = [0,1,0,0,1,0,1,1]
#labels_real = [0,0,0,0,1,1,1,1]
#labels_real_test = [0,0,0,0,1,1,1,1]

#utils.gen_shapelets(real_dat,labels_real)
#utils.gen_shapelets2(real_dat,labels_real,real_dat_test,labels_real_test)
#utils.gen_shapelets2(generated_dat,generated_dat_test,generated_dat_test,labels_gen_test)









g_3 = Actor(args2['num_classes'],args2['size'],args2['gen_embedding'])

#g_opt_3 = torch.optim.RMSprop(g_3.parameters(), lr=args2['learning_rate'])
#d_3 = Critic(args2['num_classes'],args2['size'])

#d_opt_3 = torch.optim.RMSprop(d_3.parameters(), lr=args2['learning_rate'])

    # Create Dataloader

#sines_args = args2['sines']
#arma_args = args2['arma']

#sines_args['frequency_range'][1] *= pi
#sines_args['amplitude_range'][1] *= pi

dataset = Fixed_Sines(seed = sines_args['seed'])
#train_labels_sines = torch.zeros(int(dataset.dataset.shape[0] / 2))
#train_labels = torch.tensor([])
#for i in range(NUM_OF_CLASSES):
#    train_labels = torch.cat([train_labels, torch.tensor([i] * (int(sines_args['n_series']/NUM_OF_CLASSES)))])
#train_set = [
#    (dataset.dataset[i], train_labels[i]) for i in range(len(dataset.dataset))
#]
print('Composed Dataset Chosen')
#dataloader = DataLoader(train_set, batch_size=args2['batch_size'])

    # Instantiate Trainer
#trainer_comp_3 = Trainer(g_3, d_3, g_opt_3, d_opt_3,args2['gp_weight'], args2['critic_iterations'],
#                  args2['print_every'], torch.cuda.is_available(), args2['checkpoint_frequency'])
    # Train model
#print('Training 3 is about to start...')

#trainer_comp_3.train(dataloader, epochs=args2['epochs'], plot_training_samples=True, checkpoint=args2['checkpoint'])

path = os.path.join('checkpoints','_Aug_30_2021_14_14_56_','18800_epoch.pkl')
g = utils.load_generator(path,g_3)
g = g.cuda()

data = utils.sample_generator(g,(5,50),torch.tensor([0,1,2,3,4]).long(),True)
data = data.cpu().detach().numpy()
data_real = dataset.dataset

for i in range(data.shape[0]):
    fig, ax = plt.subplots()
    ax.plot(data_real[i*40].T, label=f'Class {i} Real')
    ax.plot(data[i].T, label=f'Class {i} Fake')
    leg = ax.legend();
    ls_path = os.path.join('results', f'Class_{i}_Comparison.png')
    plt.savefig(ls_path)
    plt.show()
    plt.close()
data_real = [data_real[0], data_real[90], data_real[130], data_real[180]]
utils.gen_shapelets(data_real, [1,2,3,4], data[1:], [1,2,3,4])
