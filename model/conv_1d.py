# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:53:02 2022

@author: hlinl
"""


import os
import sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torch.utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from PARAM_ML import *


class Conv_1d_neural_net(nn.Module):
    """
    """
    
    def __init__(self, in_dim, out_dim, isConv, conv_layer_num, 
                 mlp_layer_num):
        """
        Always conv first, then MLP. Must contain MLP. 
        """
        
        self._isConv = isConv
        self.input_dim = in_dim
        self.output_dim = out_dim
        
        self.conv_layer_num = conv_layer_num
        self._conv_kernel_size = KERNEL_SIZE
        self._is_convPooling = IS_CONVPOOLING
        self._conv_padding = 0
        self._conv_stride = 1
        self._conv_pooling_kernel_size = CONV_POOLING_KERNEL_SIZE
        self._conv_pooling_padding = CONV_POOLING_PADDING
        self._conv_pooling_stride = CONV_POOLING_STRIDE
        self._conv_poolingLayer = CONV_POOLING_LAYER
        self._conv_dilation = 1
        self._conv_channel_num_init = CONV_CHANNEL_NUM_INIT
        self._conv_channel_num_mul = CONV_CHANNEL_NUM_MUL
        self._conv_predef_channel_num_list = CONV_PREDEF_CHANNEL_NUM_LIST
        self._is_convBatchNorm = IS_CONV_BATCHNORM
        self._is_convDropout = IS_CONV_DROPOUT
        self._conv_dropout_ratio = CONV_DROPOUT_RATIO # Default: 0.5. 
        
        self.mlp_layer_num = mlp_layer_num
        self._is_mlpBatchNorm = IS_MLP_BATCHNORM
        self._is_mlpDropout = IS_MLP_DROPOUT
        self._mlp_dropout_ratio = MLP_DROPOUT_RATIO # Default: 0.5.
        self._mlp_1st_layer_num = MLP_FIRST_LAYER_NUM
        self._mlp_layer_num_decay_div = MLP_LAYER_NUM_DECAY_DIV
        
        self._activationLayer = ACTIVATION_LAYER
        
        self._conv_net = None
        self._mlp_net = None
        
    
    def _convLayers_channelNumSetup(self, multiplier=2, init_outChannel_num=10, 
                                    predefined_list=[]):
        """
        """
        
        if predefined_list == []:
            channel_num_list = [1]
            
            for i in range(self.conv_layer_num):
                channel_num_list.append(init_outChannel_num*multiplier**i)
        
            return channel_num_list

        else: return predefined_list
    
    
    def _getOutputSize_from_Conv(self, in_dim, conv_kernel_size, conv_padding, 
                                 conv_stride):
        """
        """
        
        return int((in_dim + 2*conv_padding - conv_kernel_size) / conv_stride + 1)
    

    def _getOutputSize_from_Pooling(self, in_dim, pooling_kernel_size, pooling_padding, 
                                    pooling_stride):
        """
        """
        
        return int((in_dim + 2*pooling_padding - pooling_kernel_size) / pooling_stride + 1)
    
    
    def _convLayers_construction(self, in_dim):
        """
        ->| Conv -> Pooling -> ReLU -> BatchNorm -> Dropout ->|
        """

        in_dim_temp, module = in_dim, []
        conv_channel_num_list = self._convLayers_channelNumSetup(self._conv_channel_num_mul, 
                                                                 self._conv_channel_num_init,
                                                                 self._conv_predef_channel_num_list)
        
        for i in range(self.conv_layer_num):
            in_channel_temp = conv_channel_num_list[i]
            out_channel_temp = conv_channel_num_list[i+1]
            
            if i == self.conv_layer_num - 1: kernel_size_temp = in_dim_temp
            else: kernel_size_temp = self._conv_kernel_size
            
            module.append(nn.Conv1d(in_channel_temp, out_channel_temp, kernel_size_temp,
                                    stride=self._conv_stride, padding=self._conv_padding,
                                    dilation=self._conv_dilation))
            
            if i != self.conv_layer_num - 1 and self._is_convPooling: 
                module.append(self._conv_poolingLayer)
            
            module.append(self._activationLayer)
            
            if self._is_convBatchNorm: module.append(nn.BatchNorm1d(out_channel_temp))
            if self._is_convDropout: module.append(nn.Dropout(self._conv_dropout_ratio))
            
            if i != self.conv_layer_num - 1 and self._is_convPooling:
                self._getOutputSize_from_Pooling(self._getOutputSize_from_Conv(in_dim_temp, kernel_size_temp, 
                                                                               self._conv_padding, self._conv_stride), 
                                                 self._conv_pooling_kernel_size, self._conv_pooling_padding, 
                                                 self._conv_pooling_stride)
            else: 
                in_dim_temp = self._getOutputSize_from_Conv(in_dim_temp, kernel_size_temp, 
                                                            self._conv_padding, self._conv_stride)
                
        self.conv_net = nn.Sequential(*module)
        
        return out_channel_temp*in_dim_temp
        
    
    def _convLayers_construction_manual(self, in_dim):
        """
        """
        
        pass
        

    def _MLP_construction(self, in_dim):
        """
        """
        
        module = []
        in_dim_temp, out_dim_temp = in_dim, self._mlp_1st_layer_num
        
        for i in range(self.mlp_layer_num):
            if i == self.mlp_layer_num - 1: # Output layer. 
                module.append(nn.Linear(in_dim_temp, self.output_dim))
            else: 
                module.append(nn.Linear(in_dim_temp, out_dim_temp))
                module.append(self._activationLayer)
                
                if self._is_mlpBatchNorm: 
                    module.append(nn.BatchNorm1d(out_dim_temp))
                if self._is_mlpDropout:
                    module.append(nn.Dropout(self._mlp_dropout_ratio))
                
                in_dim_temp = out_dim_temp
                out_dim_temp /= self._mlp_layer_num_decay_div
        
        self._mlp_net = nn.Sequential(*module)
    
    
    def _neural_net_construction(self):
        """
        """
        
        if self._isConv:
            self._MLP_construction(self._convLayers_construction(self.input_dim))
        else: self._MLP_construction(self.input_dim)
        
    
    def forward(self, x):
        """
        """
        
        if self._isConv:
            output = self._mlp_net(x)
            output = output.view(output.size(0), -1)
        else:
            f = self.conv_net(x)
            f_flat = f.view(f.size(0), -1)
            output = self.MLP_net(f_flat)
        
        return output