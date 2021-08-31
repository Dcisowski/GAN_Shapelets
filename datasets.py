# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 23:51:13 2021

@author: dciso
"""
from typing import Sequence

from torch.utils.data import Dataset
import numpy as np
import math

class Sines(Dataset):

    def __init__(self, frequency_range: Sequence[float], amplitude_range: Sequence[float],
                 n_series: int = 200, datapoints: int = 100, seed: int = None):
        """
        Pytorch Dataset to produce sines.
        y = A * sin(B * x)
        :param frequency_range: range of A
        :param amplitude_range: range of B
        :param n_series: number of sines in your dataset
        :param datapoints: length of each sample
        :param seed: random seed
        """
        self.n_series = n_series
        self.datapoints = datapoints
        self.seed = seed
        self.frequency_range = frequency_range
        self.amplitude_range = amplitude_range
        self.dataset = self._generate_sines()

    def __len__(self):
        return self.n_series

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _generate_sines(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        x = np.linspace(start=0, stop=2 * np.pi, num=self.datapoints)
        low_freq, up_freq = self.frequency_range[0], self.frequency_range[1]
        low_amp, up_amp = self.amplitude_range[0], self.amplitude_range[1]

        freq_vector = (up_freq - low_freq) * np.random.rand(self.n_series, 1) + low_freq
        ampl_vector = (up_amp - low_amp) * np.random.rand(self.n_series, 1) + low_amp

        return ampl_vector * np.sin(freq_vector * x)
    
from statsmodels.tsa.arima_process import arma_generate_sample

class ARMA(Dataset):

    def __init__(self, AR: Sequence[float], MA: Sequence[float], seed: int = None,
                 n_series: int = 200, datapoints: int = 100):
        """
        Pytorch Dataset to sample a given ARMA process.
        
        y = ARMA(p,q)
        :param p: AR parameters
        :param q: MA parameters
        :param seed: random seed
        :param n_series: number of ARMA samples in your dataset
        :param datapoints: length of each sample
        """
        self.p = AR
        self.q = MA
        self.n_series = n_series
        self.datapoints = datapoints
        self.seed = seed
        self.dataset = self._generate_ARMA()

    def __len__(self):
        return self.n_series

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _generate_ARMA(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        ar = np.array(self.p)
        ma = np.array(self.q)
        ar = np.r_[1, -ar]
        ma = np.r_[1, ma]
        burn = int(self.datapoints / 10)

        dataset = []

        for i in range(self.n_series):
            arma = arma_generate_sample(ar=ar, ma=ma, nsample=self.datapoints, burnin=burn)
            dataset.append(arma)

        return np.array(dataset)
    
class Composed(Dataset):

    def __init__(self, sines_dict: dict, arma_dict: dict):
        """
        Pytorch Dataset to sample a given ARMA process.
        
        y = ARMA(p,q)
        :param p: AR parameters
        :param q: MA parameters
        :param seed: random seed
        :param n_series: number of ARMA samples in your dataset
        :param datapoints: length of each sample
        """
        self.sines = Sines(**sines_dict)
        self.arma = ARMA(**arma_dict)
        self.n_series = sines_dict['n_series'] + arma_dict['n_series']
        self.seed = sines_dict['seed']
        self.dataset = self._generate_comp()

    def __len__(self):
        return self.n_series

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _generate_comp(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        dataset = np.concatenate((self.sines.dataset, self.arma.dataset))
        
        return dataset
    
class Fixed_Sines(Dataset):

    def __init__(self, n_series: int = 200, datapoints: int = 100, seed: int = None):
        """
        Pytorch Dataset to produce sines.
        y = A * sin(B * x)
        :param frequency_range: range of A
        :param amplitude_range: range of B
        :param n_series: number of sines in your dataset
        :param datapoints: length of each sample
        :param seed: random seed
        """
        self.n_series = n_series
        self.datapoints = datapoints
        self.seed = seed
        self.frequencies = [2,4,7,13,16]
        self.amplitudes = [0.5,1,1.5,2.5,4.5]
        self.dataset = self._generate_sines()

    def __len__(self):
        return self.n_series

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _generate_sines(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        x = np.linspace(start=0, stop=2 * np.pi, num=self.datapoints)
        class_n_series = math.floor(self.n_series / len(self.frequencies))
        dataset = np.array([[]])
        
        freq = self.frequencies[0]
        amp = self.amplitudes[0]
    
        freq_vector = np.full((int(self.n_series/5), 1), freq, dtype=np.float64) #+ low_freq
        ampl_vector = np.full((int(self.n_series/5), 1), amp, dtype=np.float64) #+ low_amp
        for i in range(len(self.frequencies) - 1):
            freq = self.frequencies[i]
            amp = self.amplitudes[i]
    
            freq_vector = np.concatenate((freq_vector,
                                          np.full((int(self.n_series/5), 1), freq, dtype=np.float64))) #+ low_freq
            ampl_vector = np.concatenate((ampl_vector,
                                          np.full((int(self.n_series/5), 1), amp, dtype=np.float64))) #+ low_amp
        dataset = ampl_vector * np.sin(freq_vector * x)
        return dataset