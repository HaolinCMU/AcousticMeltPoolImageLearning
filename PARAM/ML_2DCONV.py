# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 19:20:26 2022

@author: hlinl
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils


# PATHS & DIRECTORIES. 
INPUT_WAVELET_SHORT_DIR = "F:/data/processed/acoustic/wavelet_spectrums_short"
OUTPUT_VISUAL_DIR = "F:/data/processed/visual"
MODEL_ARXIV_DIR = "model_checkpoints_conv2d"
TRAINING_LOG_SAVEPATH = "Train_Valid_Loss_CNN2D.png"


# ML Param. 
TEST_LAYER_FOLDER_NAMELIST = ["Layer0213_P200_V0250_C001H001S0001",
                              "Layer0429_P100_V1200_C001H001S0001",
                              "Layer0197_P250_V1200_C001H001S0001",
                              "Layer0393_P330_V0500_C001H001S0001"] # Set as empty list if no test layer specified. 

IMG_SIZE = 256 # Assume square image. 
IN_CHANNEL_NUM = 3 # Default: 1. Can be 3. 
FIRST_CONV_CHANNEL_NUM = 64
OUTPUT_DIM = 4

CONV_KERNEL_SIZE = (3,3)
CONV_STRIDE_SIZE = (1,1)
CONV_PADDING_SIZE = 1

POOLING_KERNEL_SIZE = (3,3)
POOLING_STRIDE_SIZE = (2,2)
POOLING_PADDING_SIZE = 1
POOLING_LAYER = nn.MaxPool2d(POOLING_KERNEL_SIZE)

CONV_HIDDEN_LAYER_NUM = 5
MLP_HIDDEN_LAYER_STRUCT = [256, 64, 4]
MLP_HIDDEN_LAYER_NUM = len(MLP_HIDDEN_LAYER_STRUCT)

TRAIN_RATIO = 0.8
VALID_RATIO = 0.05
TEST_RATIO = 0.15

LEARNING_RATE = 1e-4
LEARNING_RATE_SCHEDULE_PERIOD = 10 # Default: 5. 
LEARNING_RATE_DECAY_FACTOR = 1 # Default: 1. Change it to a number within [0., 1.] to define learning rate decaying rate. 

BATCH_SIZE = 64
EPOCH_NUM = 50
LAMBDA_REGLR = 1e-5
ACTIVATION_LAYER = nn.ReLU()
LOSS_FUNC = nn.MSELoss()

MODEL_CHECKPOINT_EPOCH_NUM = 5 # The epoch number for periodically archiving the intermediate 'checkpoint' models. Default: 5. 