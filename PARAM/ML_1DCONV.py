# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:30:32 2022

@author: hlinl
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import librosa


# 1D CONV PARAMS. 

IS_CONV = True
CONV_LAYER_NUM = 6
IS_CONVPOOLING = True
IS_CONV_BATCHNORM = True
IS_CONV_DROPOUT = True
CONV_DROPOUT_RATIO = 0.5 # Default: 0.5. 
CONV_POOLING_KERNEL_SIZE = 2
CONV_POOLING_STRIDE = 2
CONV_POOLING_PADDING = 0
CONV_POOLING_LAYER = nn.AvgPool1d(CONV_POOLING_KERNEL_SIZE, CONV_POOLING_STRIDE) # Default: nn.AvgPool1d(). Can be nn.MaxPool1d()
CONV_PREDEF_CHANNEL_NUM_LIST = []
CONV_CHANNEL_NUM_INIT = 10
CONV_CHANNEL_NUM_MUL = 2
KERNEL_SIZE = 3 # Default: 3. 

MLP_LAYER_NUM = 2
IS_MLP_BATCHNORM = False
IS_MLP_DROPOUT = True
MLP_DROPOUT_RATIO = 0.5
MLP_FIRST_LAYER_NUM = 512
MLP_LAYER_NUM_DECAY_DIV = 2