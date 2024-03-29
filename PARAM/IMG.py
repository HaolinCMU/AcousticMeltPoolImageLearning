# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 01:19:34 2022

@author: hlinl
"""


import os
import re
import numpy as np

from PARAM.ACOUSTIC import * 


def extract_process_param_fromImageFoldername(file_name, delimiter='_'):
    """
    """

    item_list = file_name.split(delimiter)
    P = list(map(int, re.findall(r'\d+', item_list[1])))[0] # Depends on the format of file name string. 
    V = list(map(int, re.findall(r'\d+', item_list[2])))[0] # Depends on the format of file name string. 

    return P, V


INTENSITY_THRESHOLD = (0.8, 1.0) # Default: (0.8, 1.0). (0.6, 1.0).  The intensity threshold of melt pool & bright spatters. 
SIDEBAR_COLUMNS = [(0,16), (495,511)] # The column range of side bars. Default: [(0, 32), (479, 511)]. 
SIDEBAR_THRESHOLD = 0.05
PLUME_THRESHOLD = (0.05, 0.4) # Default: (0.05, 0.4). The intensity threshold of plume and part of the melted track. 

IMAGE_SIZE = [512, 512] # [h, w]. The eventual size of processed image. Typically stay unchanged. We can change the input size right before training. 
IS_BINARY = False # Default: True. Whether to binarize the eventual processed frame. 

# DBSCAN PARAM. 
DBSCAN_EPSILON = 2
DBSCAN_MIN_PTS = 5

# PCA PARAM - one frame. 
PC_NUM_FRAME = 2
PCA_MODE_FRAME = 'transpose'
IMG_STRAIGHTEN_KEYWORD = 'plume' # Default: total. Can be one of the following: 'original', 'meltpool', 'total', 'spatters', 'plume' or 'other'. 
FRAME_ALIGN_MODE = 'principal' # 'principal' or 'secondary' or 'other'. 
FRAME_REALIGN_AXIS_VECT = np.array([0., 1.])

IMAGE_EXTENSION = "png"
HU_MOMENTS_FEATURE_IND_LIST = [0,1,2,3,4,5,6] # Must be sorted. 

# VISUAL DATA PARAM.
IMAGE_SAMPLING_RATE = 22000 # Presumed default: 22500 Hz. Estimated practical freq.: 22000 Hz. 
IMAGE_WINDOW_SIZE = int(IMAGE_SAMPLING_RATE*AUDIO_CLIP_LENGTH_DP/AUDIO_SAMPLING_RATE)
IMAGE_STRIDE_SIZE = int(IMAGE_SAMPLING_RATE*AUDIO_CLIP_STRIDE_DP/AUDIO_SAMPLING_RATE)

SELECTED_VISUAL_DATA = [('median', 'meltpool_area'),
                        ('std', 'meltpool_area'), 
                        ('median', 'P'), 
                        ('std', 'V')] # keyword_list = [(`featurization_mode`, `feature_type`)]
VISUAL_DATA_FEATURIZATION_MODE = list(set([keys[0] for keys in SELECTED_VISUAL_DATA])) # Default: 'median'. Options: 'mean', 'median', 'std', and 'sliding'. 
VISUAL_DATA_FEATURE_LIST = list(set([keys[1] for keys in SELECTED_VISUAL_DATA])) # P and V are always contained in the end of the feature list. 

VISUAL_DATA_EXTENSION = "npy"

IS_STANDARD = True

