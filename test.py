# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:11:11 2022

@author: hlinl
"""


import os
import sys
DIR_ABS = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR_ABS))
import time

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from torch.utils.data import Dataset, DataLoader

from PARAM import *
from model import *

from dataset import *
from files import *


class Validation_VAE(object):
    """
    """

    def __init__(self, trained_model_path, model_class, input_data_dir, output_data_dir, dataset_class, loss_class,
                 loss_beta=1., model=None, test_dataloader=None, batch_size=1):
        """
        """

        self.model_path = trained_model_path
        self.model_class = model_class
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.dataset_class = dataset_class
        self.loss_class = loss_class
        self.loss_beta = loss_beta
        self.test_dataloader = test_dataloader
        self.batch_size = batch_size

        self._is_cuda = torch.cuda.is_available() # Bool. Indicate whether an available cuda is installed. 
        self._device = torch.device('cuda' if self._is_cuda else 'cpu') # Torch.device. Indicate 'cuda' or 'cpu'. 
        # self._device = torch.device('cpu') # Torch.device. 'cpu' only. 

        self.model = self.model_class().load_state_dict(torch.load(self.model_path)) if model is None else model
        self.model.to(self._device)
        # self.model.load_state_dict(torch.load(self.model_path))

        self.dataset = self.dataset_class(self.input_data_dir, self.output_data_dir)
        self.dataset_size = len(self.dataset)

        self._loss_func = self.loss_class(loss_beta=self.loss_beta) # Torch.nn.Module. Self-defined loss function for VAE neural network. 

        self.init_dataloader()

    
    def init_dataloader(self):
        """
        """

        if self.test_dataloader is None:
            dataset_indices = list(range(self.dataset_size))
            sampler = torch.utils.data.SubsetRandomSampler(dataset_indices)
            self.test_dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler)

        else: pass

    
    def evaluate(self):
        """
        """

        loss_list, groundtruths_list, generations_list, latent_list = [], [], [], []
        
        for _, batch in enumerate(self.test_dataloader): 
            inputs_test, groundtruths_test = copy.deepcopy(batch['input']), copy.deepcopy(batch['output'])
            batch_size_temp, c, h, w = groundtruths_test.size()
            
            inputs_test = inputs_test.to(self._device)
            groundtruths_test = groundtruths_test.to(self._device)
            
            self.model.eval()
            with torch.no_grad():
                y, latent = self.model(inputs_test)
                loss_test_perBatch = self._loss_func(y, groundtruths_test)

                loss_list.append(float(loss_test_perBatch.item()/batch_size_temp)) # Loss per test sample.
                if len(groundtruths_list) <= 50: # Prescribe size limit to save space. 
                    groundtruths_list += [groundtruths_test.cpu().numpy()[i,:].reshape(c, h, w) for i in range(batch_size_temp)]
                if len(generations_list) <= 50: # Prescribe size limit to save space. 
                    generations_list += [y.cpu().numpy()[i,:].reshape(c, h, w) for i in range(batch_size_temp)]
                latent_list += [latent.cpu().numpy()[i,:] for i in range(batch_size_temp)]
        
        return loss_list, groundtruths_list, generations_list, latent_list
    

    def showcase_generation(self, ind=None):
        """
        """

        pass


    def sample_from_latent(self):
        """
        """

        pass


    def generate(self, latent):
        """
        """

        pass


if __name__ == "__main__":
    """
    """

    pass