"""
Created on Tue Aug  2 23:35:01 2022

@author: hlinl
"""


import os
import gc
import glob
import sys
import time
DIR_ABS = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR_ABS))

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt

from torch.utils.data import Dataset, DataLoader

from PARAM import *


class CNN_2D(nn.Module):
    """
    A simple parameterized 2D CNN model generator. 
    """

    def __init__(self, in_channel_num, output_dim, conv_hidden_layer_num, mlp_hidden_layer_struct=ML_2DCONV.MLP_HIDDEN_LAYER_STRUCT):
        """
        Conv block first, then MLP block. 
        """
        
        super(CNN_2D, self).__init__()
        self.in_channel_num = in_channel_num
        self.output_dim = output_dim
        self.conv_hidden_layer_num = conv_hidden_layer_num
        self.mlp_hidden_layer_struct = mlp_hidden_layer_struct
        self.mlp_hidden_layer_num = len(self.mlp_hidden_layer_struct)
        
        self._conv_net = None
        self._mlp_net = None

        self._is_conv_net = True if self.conv_hidden_layer_num != 0 else False
        self._is_mlp_net = True if self.mlp_hidden_layer_num != 0 else False

        self._build()


    @property
    def conv_net(self):
        """
        """

        if self._conv_net is not None: return self._conv_net
        else: raise ValueError("Convolution net not created. ")
    

    @property
    def mlp_net(self):
        """
        """

        if self._mlp_net is not None: return self._mlp_net
        else: raise ValueError("MLP net not created. ")
    

    def forward(self, x):
        """
        """

        if self._conv_net is None or not self._is_conv_net:
            raise ValueError("Neural network model not generated. ")
        
        conv_encode = self._conv_net(x)
        output = conv_encode.view(conv_encode.size(0), -1)
        if self._is_mlp_net: output = self._mlp_net(output)

        return output


    def _after_conv_img_dim(self, in_dim, conv_kernel_size, conv_padding, conv_stride):
        """
        """

        return int((in_dim + 2*conv_padding - conv_kernel_size) / conv_stride + 1)


    def _after_pool_img_dim(self, in_dim, pool_kernel_size, pool_padding, pool_stride):
        """
        """

        return int((in_dim + 2*pool_padding - pool_kernel_size) / pool_stride + 1)

    
    def _conv_block(self):
        """
        """

        architecture_pre_module = []

        for i in range(self.conv_hidden_layer_num):
            if i == 0: 
                in_channel_num_temp = self.in_channel_num
                out_channel_num_temp = ML_2DCONV.FIRST_CONV_CHANNEL_NUM
                img_size_temp = ML_2DCONV.IMG_SIZE
                output_img_dim_temp = None
            else: 
                in_channel_num_temp = copy.deepcopy(out_channel_num_temp)
                out_channel_num_temp = copy.deepcopy(int(in_channel_num_temp*2))
                img_size_temp = copy.deepcopy(output_img_dim_temp)
            
            architecture_pre_module.append(nn.Conv2d(in_channel_num_temp, out_channel_num_temp, 
                                                     kernel_size=ML_2DCONV.CONV_KERNEL_SIZE, 
                                                     stride=ML_2DCONV.CONV_STRIDE_SIZE, 
                                                     padding=ML_2DCONV.CONV_PADDING_SIZE))
            # architecture_pre_module.append(nn.Dropout(0.5)) # Dropout layer. 
            architecture_pre_module.append(ML_2DCONV.ACTIVATION_LAYER)
            architecture_pre_module.append(nn.BatchNorm2d(out_channel_num_temp))
            
            after_conv_img_dim = self._after_conv_img_dim(img_size_temp, conv_kernel_size=ML_2DCONV.CONV_KERNEL_SIZE, 
                                                          conv_padding=ML_2DCONV.CONV_PADDING_SIZE, 
                                                          conv_stride=ML_2DCONV.CONV_STRIDE_SIZE)

            if i != self.conv_hidden_layer_num - 1:
                architecture_pre_module.append(ML_2DCONV.POOLING_LAYER)
                output_img_dim_temp = self._after_pool_img_dim(after_conv_img_dim, 
                                                               pool_kernel_size=ML_2DCONV.POOLING_KERNEL_SIZE, 
                                                               pool_padding=ML_2DCONV.POOLING_PADDING_SIZE, 
                                                               pool_stride=ML_2DCONV.POOLING_STRIDE_SIZE)
            else: 
                architecture_pre_module.append(nn.AvgPool2d(after_conv_img_dim)) # Global average pooling. 
                output_img_dim_temp = self._after_pool_img_dim(after_conv_img_dim, pool_kernel_size=after_conv_img_dim, 
                                                               pool_padding=0, pool_stride=1)

        self._conv_net = nn.Sequential(*architecture_pre_module)
        out_dim = int(out_channel_num_temp * output_img_dim_temp**2)

        return out_dim # Ideally should be identical to `out_channel_num_temp`. 
    

    def _mlp_block(self, in_dim):
        """
        """

        architecture_pre_module = []

        for i in range(self.mlp_hidden_layer_num):
            if i == 0: 
                in_dim_temp = in_dim
                output_dim_temp = self.mlp_hidden_layer_struct[0]
            else: 
                in_dim_temp = copy.deepcopy(output_dim_temp)
                output_dim_temp = self.mlp_hidden_layer_struct[i]

            architecture_pre_module.append(nn.Linear(in_dim_temp, output_dim_temp))
            
            if i != self.mlp_hidden_layer_num - 1: 
                architecture_pre_module.append(ML_2DCONV.ACTIVATION_LAYER)
                architecture_pre_module.append(nn.Dropout(0.5)) # Dropout layer. 
        
        self._mlp_net = nn.Sequential(*architecture_pre_module)


    def _build(self):
        """
        """

        conv_out_dim = None

        if self._is_conv_net: 
            conv_out_dim = self._conv_block()
            if self._is_mlp_net: self._mlp_block(conv_out_dim)
        else: pass


if __name__ == "__main__":
    """
    """

    conv2d_net = CNN_2D(in_channel_num=3, output_dim=1, conv_hidden_layer_num=6, mlp_hidden_layer_struct=[128, 1])
    print(conv2d_net)
    pytorch_total_params = sum(p.numel() for p in conv2d_net.parameters() if p.requires_grad)
    print(pytorch_total_params)


