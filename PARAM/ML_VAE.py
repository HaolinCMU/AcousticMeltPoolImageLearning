# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:35:35 2021

@author: hlinl
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import librosa


# VAE & Autoencoder PARAMS. 
INPUT_IMAGE_SIZE = [512, 512] # # [h, w]. The image size of the input image. 
OUTPUT_IMAGE_SIZE = [512, 512] # # [h, w]. The image size of the output image. 

INPUT_IMAGE_DATA_DIR = "img_data_binary"
OUTPUT_IMAGE_DATA_DIR = "img_data_binary"

BATCH_SIZE = 64
NUM_EPOCHS = 20 # Default: 25. 
LOSS_BETA = 1. # Hyperparameter for controlling weights of the KL-divergence loss function term. 
LOSS_RECONSTRUCT_MODE = 'MSE' # 'MSE' (general) or 'BCE' (binarized-specific). 
LAMBDA_REGLR = 1e-5 # Set regularization in optimizer. 0 for not applying regularization. 

LEARNING_RATE = 1e-4 # Default: 1e-5. 
LEARNING_RATE_SCHEDULE_PERIOD = 10 # Default: 5. 
LEARNING_RATE_DECAY_FACTOR = 1 # Default: 1. Change it to a number within [0., 1.] to define learning rate decaying rate. 

MODEL_ARXIV_DIR = "model_checkpoints"
TRAINING_LOG_SAVEPATH = "vae_train.log"
TRAIN_VALID_LOSS_SAVEPATH = "Train_Valid_Loss_VAE.png"

ACTIVATION_LAYER = nn.LeakyReLU() # Default: nn.ReLU(). 


# ------------- Temp -------------
MLP_FIRST_LAYER_NUM = 512
MLP_LAYER_NUM_DECAY_DIV = 2
