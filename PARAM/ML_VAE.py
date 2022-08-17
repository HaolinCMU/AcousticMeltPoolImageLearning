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


# VAE & Autoencoder PARAMS. 
INPUT_IMAGE_SIZE = [512, 512] # # [h, w]. The image size of the input image. 
OUTPUT_IMAGE_SIZE = [512, 512] # # [h, w]. The image size of the output image. 

INPUT_IMAGE_DATA_DIR = "img_data_grayscale"
OUTPUT_IMAGE_DATA_DIR = "img_data_grayscale"

BATCH_SIZE = 64
NUM_EPOCHS = 50 # Default: 25. Debug: 1. Short: 5. Medium: 15 - 30. Long: 50. 
LOSS_BETA = 1e-1 # Penalty factor that controls weights of the KL-divergence loss function term. Used only for VAE. 
LOSS_RECONSTRUCT_MODE = 'MSE' # 'MSE' (general) or 'BCE' (binarized-specific). 
LAMBDA_REGLR = 1e-5 # Set regularization in optimizer. 0 for not applying regularization. 

LEARNING_RATE = 1e-4 # Default: 1e-5. 
LEARNING_RATE_SCHEDULE_PERIOD = 10 # Default: 5. 
LEARNING_RATE_DECAY_FACTOR = 1 # Default: 1. Change it to a number within [0., 1.] to define learning rate decaying rate. 

MODEL_ARXIV_DIR = "model_checkpoints_autoencoder"
TRAINING_LOG_SAVEPATH = "vae_train.log"
TRAIN_VALID_LOSS_SAVEPATH = "Train_Valid_Loss_VAE.png"

ACTIVATION_LAYER = nn.LeakyReLU() # Default: nn.ReLU(). 

LATENT_DIM = 32 # Default: 32. 
BOTTLENECK_DIM = 1024 # Default: 512. 

MODEL_CHECKPOINT_EPOCH_NUM = 5 # The epoch number for periodically archiving the intermediate 'checkpoint' models. Default: 5. 

# ------------- Temp -------------
MLP_FIRST_LAYER_NUM = 512
MLP_LAYER_NUM_DECAY_DIV = 2
