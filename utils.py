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
    with open("args2.yaml", 'r') as stream:
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

def gen_shapelets(data_real, labels_real, data_fake, labels_fake):
    
    numpy.random.seed(42)

    # Load the Trace dataset
    X_train_real = data_real
    y_train_real = labels_real
    
    X_train_fake = data_fake
    y_train_fake = labels_fake
    
    # Normalize the time series
    X_train_real = TimeSeriesScalerMinMax().fit_transform(X_train_real)
    
    X_train_fake = TimeSeriesScalerMinMax().fit_transform(X_train_fake)
    
    # Get statistics of the dataset
    n_ts_real, ts_sz_real = X_train_real.shape[:2]
    n_classes_real = len(set(y_train_real))
    
    n_ts_fake, ts_sz_fake = X_train_fake.shape[:2]
    n_classes_fake = len(set(y_train_fake))
    
    # We will extract 2 shapelets and align them with the time series
    shapelet_sizes = {100: 4}
    
    # Define the model and fit it using the training data
    shp_clf_real = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                weight_regularizer=0.0001,
                                optimizer=Adam(lr=0.01),
                                max_iter=300,
                                verbose=0,
                                scale=False,
                                random_state=42)
    
    shp_clf_fake = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                weight_regularizer=0.0001,
                                optimizer=Adam(lr=0.01),
                                max_iter=300,
                                verbose=0,
                                scale=False,
                                random_state=42)
    shp_clf_real.fit(X_train_real, y_train_real)
    #shp_clf_fake.fit(X_train_fake, y_train_fake)
    
    # We will plot our distances in a 2D space
    distances_real = shp_clf_real.transform(X_train_real).reshape((-1, 2))
    weights_real, biases_real = shp_clf_real.get_weights('classification')
    
#    distances_fake = shp_clf_fake.transform(X_train_real).reshape((-1, 2))
#    weights_fake, biases_fake = shp_clf_fake.get_weights('classification')
    
    # Create a grid for our two shapelets on the left and distances on the right
    viridis = cm.get_cmap('viridis', 4)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 9)
    fig_ax1 = fig.add_subplot(gs[0, :2])
    fig_ax2 = fig.add_subplot(gs[0, 2:4])
    fig_ax3 = fig.add_subplot(gs[1, :2])
    fig_ax4 = fig.add_subplot(gs[1, 2:4])
    fig_ax1a = fig.add_subplot(gs[0, 4:6])
    fig_ax2a = fig.add_subplot(gs[0, 6:8])
    fig_ax3a = fig.add_subplot(gs[1, 4:6])
    fig_ax4a = fig.add_subplot(gs[1, 6:8])
    
    # Plot our two shapelets on the left side
    fig_ax1.plot(shp_clf_real.shapelets_[0])
    fig_ax1.set_title('Shapelet $\mathbf{sr}_1$')
    
    fig_ax1a.plot(data_fake[0].T)
    fig_ax1a.set_title('Shapelet $\mathbf{sf}_1$')
    
    fig_ax2.plot(shp_clf_real.shapelets_[1])
    fig_ax2.set_title('Shapelet $\mathbf{sr}_2$')
    
    fig_ax2a.plot(data_fake[1].T)
    fig_ax2a.set_title('Shapelet $\mathbf{sf}_2$')
    
    fig_ax3.plot(shp_clf_real.shapelets_[2])
    fig_ax3.set_title('Shapelet $\mathbf{sr}_3$')
    
    fig_ax3a.plot(data_fake[2].T)
    fig_ax3a.set_title('Shapelet $\mathbf{sf}_3$')
    
    fig_ax4.plot(shp_clf_real.shapelets_[3])
    fig_ax4.set_title('Shapelet $\mathbf{sr}_4$')
    
    fig_ax4a.plot(data_fake[3].T)
    fig_ax4a.set_title('Shapelet $\mathbf{sf}_4$')
    
    # Create the time series of each class
    #for i, subfig in enumerate([fig_ax3a, fig_ax3b, fig_ax3c, fig_ax3d]):
    #    for k, ts in enumerate(X_train[y_train == i + 1]):
    #        subfig.plot(ts.flatten(), c=viridis(i / 3), alpha=0.25)
    #        subfig.set_title('Class {}'.format(i + 1))
    #fig.text(x=.15, y=.02, s='Input time series', fontsize=12)
    
    # Create a scatter plot of the 2D distances for the time series of each class.
    #distances = distances[:len(y_train_real)]
    #for i, y in enumerate(numpy.unique(y_train_real)):
    #    fig_ax4.scatter(distances[y_train_real == y][:, 0],
    #                    distances[y_train_real == y][:, 1],
    #                    c=[viridis(i / 3)] * numpy.sum(y_train_real == y),
    #                    edgecolors='k',
    #                    label='Class {}'.format(y))
    
    # Create a meshgrid of the decision boundaries
    #xmin = numpy.min(distances[:, 0]) - 0.1
    #xmax = numpy.max(distances[:, 0]) + 0.1
    #ymin = numpy.min(distances[:, 1]) - 0.1
    #ymax = numpy.max(distances[:, 1]) + 0.1
    #xx, yy = numpy.meshgrid(numpy.arange(xmin, xmax, (xmax - xmin)/200),
    #                        numpy.arange(ymin, ymax, (ymax - ymin)/200))
    #Z = []
    #for x, y in numpy.c_[xx.ravel(), yy.ravel()]:
    #    Z.append(numpy.argmax([biases[i] + weights[0][i]*x + weights[1][i]*y
    #                           for i in range(4)]))
    #Z = numpy.array(Z).reshape(xx.shape)
    #cs = fig_ax4.contourf(xx, yy, Z / 3, cmap=viridis, alpha=0.25)
    
    path = os.path.join('results', 'shapelets_comparison.jpg')
    plt.savefig(path)
   
    plt.show()
    
def gen_shapelets2(data_train, labels_train, data_test, labels_test):
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