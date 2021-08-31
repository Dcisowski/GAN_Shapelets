# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:58:38 2021

@author: dciso
"""
import argparse
import os
import os.path

import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
from torch.utils.tensorboard import writer, SummaryWriter
from torch.utils.data import DataLoader
from math import pi

from datetime import date, datetime

from datasets import Sines
#from model import Actor, Critic
import utils


class Trainer:
    NOISE_LENGTH = 50

    def __init__(self, generator, critic, gen_optimizer, critic_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=200, use_cuda=False, checkpoint_frequency=200):
        self.g = generator
        self.g_opt = gen_optimizer
        self.c = critic
        self.c_opt = critic_optimizer
        self.losses = {'g': [], 'c': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.checkpoint_frequency = checkpoint_frequency
        if self.use_cuda:
            self.g.cuda()
            self.c.cuda()

    def _critic_train_iteration(self, real_data, condition, labels):

        batch_size = real_data.size()[0]
        noise_shape = (batch_size, self.NOISE_LENGTH)
        generated_data = utils.sample_generator(self.g, noise_shape, labels, self.use_cuda)

        real_data = Variable(real_data)

        if self.use_cuda:
            real_data = real_data.cuda()
            labels = labels.cuda()

        # Pass data through the Critic
        c_real = self.c(real_data,labels)
        c_generated = self.c(generated_data,labels)

        # Get gradient penalty
        gradient_penalty = utils._gradient_penalty(real_data, generated_data, self.c,
                                                   self.losses, self.gp_weight, condition,
                                                   labels, self.use_cuda)

        # Create total loss and optimize
        self.c_opt.zero_grad()
        d_loss = c_generated.mean() - c_real.mean() + gradient_penalty
        d_loss.backward()
        self.c_opt.step()

        if condition:
            self.losses['GP'].append(gradient_penalty.data.item())
            self.losses['c'].append(d_loss.data.item())

    def _generator_train_iteration(self, data, condition, labels):
        self.g_opt.zero_grad()
        batch_size = data.size()[0]
        latent_shape = (batch_size, self.NOISE_LENGTH)

        generated_data = utils.sample_generator(self.g, latent_shape, labels, self.use_cuda)

        # Calculate loss and optimize
        if self.use_cuda:
            labels = labels.cuda()
        d_generated = self.c(generated_data, labels)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.g_opt.step()
        #self.losses['g'].append(g_loss.data.item())
        return g_loss.data.item()

    

    def _train_epoch(self, data_loader, epoch):
        dt_len = len(data_loader)
        g_loss = 0
        for i, (data, labels) in enumerate(data_loader):
            self.num_steps += 1
            labels = labels.long()
            if self.use_cuda: 
                labels.cuda()
                data.cuda()
            self._critic_train_iteration(data.float(), i == dt_len - 1, labels)
            # Only update generator every critic_iterations iterations
            if self.num_steps % self.critic_iterations == 0:
                g_loss = self._generator_train_iteration(data, i == dt_len - 1, labels)

            if i % self.print_every == 0:
                global_step = i + epoch * len(data_loader.dataset)
                
        
        self.losses['g'].append(g_loss)
            
    def train(self, data_loader, epochs, plot_training_samples=True, checkpoint=None):

        if checkpoint:
            path = os.path.join('checkpoints', checkpoint['datetime'], checkpoint['epoch'])
            state_dicts = torch.load(path, map_location=torch.device('cpu'))
            self.g.load_state_dict(state_dicts['g_state_dict'])
            self.c.load_state_dict(state_dicts['d_state_dict'])
            self.g_opt.load_state_dict(state_dicts['g_opt_state_dict'])
            self.c_opt.load_state_dict(state_dicts['d_opt_state_dict'])
            
       
        # Define noise_shape
        noise_shape = (1, self.NOISE_LENGTH)
        
        # Create folder for storing losses and latents plots as well as 
        # folder for storing checkpoints
        _date = f'_{date.today().strftime("%b_%d_%Y")}_{datetime.now().strftime("%H_%M_%S")}_'
        
        losses_path = os.path.join('training_samples', 'Losses', _date)
        #fixed_latents_path = os.path.join('training_samples', 'fixed_latents', _date)
        #dynamic_latents_path = os.path.join('training_samples', 'dynamic_latents', _date)
        checkpoints_path = os.path.join('checkpoints', _date)
        
        os.mkdir(losses_path)
            #os.mkdir(fixed_latents_path)
            #os.mkdir(dynamic_latents_path)
        os.mkdir(checkpoints_path)
        
        #if self.use_cuda: data_loader.cuda()
            
        if plot_training_samples:
            # Fix latents to see how series generation improves during training
            fixed_latents = Variable(utils.sample_latent(noise_shape))
            if self.use_cuda:
                fixed_latents = fixed_latents.cuda()

        for epoch in tqdm(range(epochs)):

            # Sample a different region of the latent distribution to check for mode collapse
            dynamic_latents = Variable(utils.sample_latent(noise_shape))
            if self.use_cuda:
                dynamic_latents = dynamic_latents.cuda()

            self._train_epoch(data_loader, epoch + 1)

            # Save checkpoint
            if epoch % self.checkpoint_frequency == 0:
                cp_path = os.path.join(checkpoints_path, f'{epoch}_epoch.pkl')
                torch.save({
                    'epoch': epoch,
                    'd_state_dict': self.c.state_dict(),
                    'g_state_dict': self.g.state_dict(),
                    'd_opt_state_dict': self.c_opt.state_dict(),
                    'g_opt_state_dict': self.g_opt.state_dict(),
                },cp_path)

            if plot_training_samples and (epoch % self.print_every == 0) :
                self.g.eval()
                #Generate fake data using both fixed and dynamic latents
#                fake_data_fixed_latents = self.g(fixed_latents).cpu().data
 #               fake_data_dynamic_latents = self.g(dynamic_latents).cpu().data

                #plt.figure()
                #plt.plot(fake_data_fixed_latents.numpy()[0].T)
                #fl_path = os.path.join(fixed_latents_path, f'{epoch}_epoch.png')
                #plt.savefig(fl_path)
                #plt.close()

                #plt.figure()
                #plt.plot(fake_data_dynamic_latents.numpy()[0].T)
                #dl_path = os.path.join(dynamic_latents_path, f'{epoch}_epoch.png')
                #plt.savefig(dl_path)
                #plt.close()
                
                fig, ax = plt.subplots()
                ax.plot(self.losses['c'], label='Critic')
                ax.plot(self.losses['GP'], label='Gradient Penalty')
                ax.plot(self.losses['gradient_norm'], label='Gradient Norm')
                ax.plot(self.losses['g'], label='Generator')
                leg = ax.legend();
                ls_path = os.path.join(losses_path, f'{epoch}_epoch.png')
                plt.savefig(ls_path)
                plt.close()
                
                self.g.train()


    #@staticmethod
    #def sample_latent(shape):
    #    return torch.randn(shape)

    


    
    
