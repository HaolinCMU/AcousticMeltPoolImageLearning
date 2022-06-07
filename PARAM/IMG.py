# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 01:19:34 2022

@author: hlinl
"""

import os
import numpy as np

INTENSITY_THRESHOLD = (0.6, 1.0) # Default: (0.8, 1.0). 
IMAGE_SIZE = [512, 512] # [h, w]. The eventual size of processed image. Typically stay unchanged. We can change the input size right before training. 
IS_BINARY = True # Default: True. Whether to binarize the eventual processed frame. 

# DBSCAN PARAM
DBSCAN_EPSILON = 2
DBSCAN_MIN_PTS = 5

# PCA PARAM - one frame. 
PC_NUM_FRAME = 2
PCA_MODE_FRAME = 'transpose'
IMG_STRAIGHTEN_KEYWORD = 'total' # Default: total. Can be one of the following: 'meltpool', 'total', 'spatters' or 'other'. 
FRAME_ALIGN_MODE = 'principal' # 'principal' or 'secondary' or 'other'. 
FRAME_REALIGN_AXIS_VECT = np.array([0.,1.])

IMAGE_EXTENSION = "png"
HU_MOMENTS_FEATURE_IND_LIST = [0,1,2,3,4,5,6] # Must be sorted. 