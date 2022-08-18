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
INPUT_WAVELET_SHORT_DIR = "C:/Users/hlinl/Desktop/acoustic_image_learning/data/new_dataset/wavelet_spectrums_short" # "F:/data/processed/acoustic/wavelet_spectrums_short"
OUTPUT_VISUAL_DIR = "C:/Users/hlinl/Desktop/acoustic_image_learning/data/new_dataset/visual" # "F:/data/processed/visual"
MODEL_ARXIV_DIR = "model_checkpoints_conv2d"
TRAINING_LOG_SAVEPATH = "cnn2d_train.log"
TRAIN_VALID_LOSS_SAVEPATH = "Train_Valid_Loss_cnn2d.png"


# ML Param. 
TEST_LAYER_FOLDER_NAMELIST = ["Layer0213_P200_V0250_C001H001S0001",
                              "Layer0429_P100_V1200_C001H001S0001",
                              "Layer0197_P250_V1200_C001H001S0001",
                              "Layer0393_P330_V0500_C001H001S0001"] # Set as empty list if no test layer specified. 
VISUAL_FEATURE_LIST = [0] # Default: [0, 1]. [0, 1, 2, 3] = [area, aspect ratio, P, V]. Depends on function `image_processing.collect_visual_data()`. 

IMG_SIZE = 256 # Assume square image. Default: 256.  
IN_CHANNEL_NUM = 3 # Default: 1. Can be 3. 
FIRST_CONV_CHANNEL_NUM = 16 # Default: 64. 
OUTPUT_DIM = len(VISUAL_FEATURE_LIST)

CONV_KERNEL_SIZE = 3 # (3,3). 
CONV_STRIDE_SIZE = 1 # (1,1). 
CONV_PADDING_SIZE = 1 # Default: 1. 

POOLING_KERNEL_SIZE = 3 # (3,3)
POOLING_STRIDE_SIZE = 2 # (2,2)
POOLING_PADDING_SIZE = 1
POOLING_LAYER = nn.MaxPool2d(kernel_size=POOLING_KERNEL_SIZE, 
                             stride=POOLING_STRIDE_SIZE, 
                             padding=POOLING_PADDING_SIZE)

CONV_HIDDEN_LAYER_NUM = 6 # 6, if `IMG_SIZE` starts from 256. 
MLP_HIDDEN_LAYER_STRUCT = [128, OUTPUT_DIM] # Default: [256, 64, OUTPUT_DIM]. 
MLP_HIDDEN_LAYER_NUM = len(MLP_HIDDEN_LAYER_STRUCT)

TRAIN_RATIO = 0.8
VALID_RATIO = 0.05
TEST_RATIO = 0.15
IS_SHUFFLE_IN_DATALOADER = True # Default: True. Shuffle the training and validation dataset separately. 
IS_RANDOM_PARTITION = True

LEARNING_RATE = 1e-4
LEARNING_RATE_SCHEDULE_PERIOD = 10 # Default: 5. 
LEARNING_RATE_DECAY_FACTOR = 0.5 #  Default: 1. A number within [0., 1.] to define learning rate decaying rate. 

BATCH_SIZE = 128 # Default: 32, If `IMG_SIZE` is 256. 
NUM_EPOCHS = 40 # Default: 20. 
LAMBDA_REGLR = 1e-5 # Default: 1e-5. 
ACTIVATION_LAYER = nn.ReLU(inplace=True)
LOSS_FUNC = nn.MSELoss()

MODEL_CHECKPOINT_EPOCH_NUM = 5 # The epoch number for periodically archiving the intermediate 'checkpoint' models. Default: 5. 