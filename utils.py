# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 00:09:31 2021

@author: dciso
"""
import yaml
import os

import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from sklearn.metrics import accuracy_score
import numpy
import math
from matplotlib import cm
import matplotlib.pyplot as plt

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import LearningShapelets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict
from tslearn.utils import ts_size
from tensorflow.keras.optimizers import Adam

#from model import Actor

def load_args():
    with open("args.yaml", 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def load_args2():
    with open("args.yaml", 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
def load_generator(path, g):
    """
        :param path: path to a pickle file
        :param g: generator
        """
    state_dicts = torch.load(path, map_location=torch.device('cpu'))
    g.load_state_dict(state_dicts['g_state_dict'])
    return g
            
class AddDimension(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)


class SqueezeDimension(nn.Module):
    def forward(self, x):
        return x.squeeze(1)
    
def plot_figures(data, n_classes):
    
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(math.ceil(n_classes*0.25), n_classes)
    figs = []
    num_of_cols = math.ceil(n_classes*0.25)
    num_of_rows = math.ceil(n_classes/num_of_cols)
    
        
    for x in data:
        print(x.shape)

def plot_results(data_real, generated_data):
    for i in range(data.shape[0]):
        fig, ax = plt.subplots()
        ax.plot(data_real[i * 40].T, label=f'Class {i} Real')
        ax.plot(generated_data[i].T, label=f'Class {i} Fake')
        leg = ax.legend()
        ls_path = os.path.join('results', f'Class_{i}_Comparison.png')
        plt.savefig(ls_path)
        plt.show()
        plt.close()
        
def _gradient_penalty(real_data, generated_data, 
                      c, losses, 
                      gp_weight, condition, 
                      labels, use_cuda):

        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        if use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if use_cuda:
            interpolated = interpolated.cuda()
            labels = labels.cuda()

        # Pass interpolated data through Critic
        prob_interpolated = c(interpolated, labels)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if use_cuda
                               else torch.ones(prob_interpolated.size()), create_graph=True,
                               retain_graph=True)[0]
        # Gradients have shape (batch_size, num_channels, series length),
        # here we flatten to take the norm per example for every batch
        gradients = gradients.view(batch_size, -1)
        if condition:
            losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data.item())

        # Derivatives of the gradient close to 0 can cause problems because of the
        # square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return gp_weight * ((gradients_norm - 1) ** 2).mean()
    
def sample_generator(g, latent_shape, labels, use_cuda):
        latent_samples = Variable(sample_latent(latent_shape))
        if use_cuda:
            latent_samples = latent_samples.cuda()
            labels = labels.cuda()

        return g(latent_samples, labels)

def sample_latent(shape):
    return torch.randn(shape)

def sample(self, num_samples):
    generated_data = self.sample_generator(num_samples)
    return generated_data.data.cpu().numpy()

    
def gen_shapelets(data_train, labels_train, data_test, labels_test):
    numpy.random.seed(0)

    # Load the Trace dataset
    X_train = data_train
    y_train = labels_train
    X_test = data_test
    y_test = labels_test
    
    # Normalize each of the timeseries in the Trace dataset
    X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
    X_test = TimeSeriesScalerMinMax().fit_transform(X_test)
    
    # Get statistics of the dataset
    n_ts, ts_sz = X_train.shape[:2]
    n_classes = len(set(y_train))
    
    # Set the number of shapelets per size as done in the original paper
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                           ts_sz=ts_sz,
                                                           n_classes=n_classes,
                                                           l=0.1,
                                                           r=1)
    
    # Define the model using parameters provided by the authors (except that we
    # use fewer iterations here)
    shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                optimizer=Adam(.01),
                                batch_size=16,
                                weight_regularizer=.01,
                                max_iter=200,
                                random_state=42,
                                verbose=0)
    shp_clf.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy score
    pred_labels = shp_clf.predict(X_test)
    print("Correct classification rate:", accuracy_score(y_test, pred_labels))
    
    # Plot the different discovered shapelets
    plt.figure()
    for i, sz in enumerate(shapelet_sizes.keys()):
        plt.subplot(len(shapelet_sizes), 1, i + 1)
        plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
        for shp in shp_clf.shapelets_:
            if ts_size(shp) == sz:
                plt.plot(shp.ravel())
        plt.xlim([0, max(shapelet_sizes.keys()) - 1])
    
    plt.tight_layout()
    plt.show()
    
    # The loss history is accessible via the `model_` that is a keras model
    plt.figure()
    plt.plot(numpy.arange(1, shp_clf.n_iter_ + 1), shp_clf.history_["loss"])
    plt.title("Evolution of cross-entropy loss during training")
    plt.xlabel("Epochs")
    plt.show()