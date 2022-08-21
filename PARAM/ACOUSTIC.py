# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:14:22 2022

@author: hlinl
"""


import os
import numpy as np
import re
import torch
import torch.nn as nn
import torch.utils

from collections import defaultdict


def set_process_param_defect_dict(process_param_defect_file):
    """
    """

    process_param_defect_dict = defaultdict(lambda: defaultdict(dict))

    data = np.genfromtxt(process_param_defect_file, delimter=',').astype(int)
    for i in range(data.shape[0]):
        P_temp, V_temp, defect_temp = data[i,:]
        process_param_defect_dict[(P_temp, V_temp)] = defect_temp
    
    return process_param_defect_dict


def get_defect_type_fromPV(P, V):
    """
    """

    pass


def extract_process_param_fromAcousticFilename(file_name, delimiter='_'):
    """
    """

    item_list = file_name.split(delimiter)
    P = list(map(int, re.findall(r'\d+', item_list[1])))[0] # Depends on the format of file name string. 
    V = list(map(int, re.findall(r'\d+', item_list[2])))[0] # Depends on the format of file name string. 

    return P, V


AUDIO_SENSOR_NO = 0 # The chosen acoustic sensor of interest. Options: [0, 1, 2]. 
PHOTODIODE_SENSOR_NO = 0 # The chosen photodiode sensor of interest. Options: [0, 1]. 
INITIAL_LAYER_NUM = 17 # The number of beginning layers of the print. 
TRANSITIONAL_LAYER_NUM = 3 # The number of transitional layers in between two consecutive layers of interest. 
REPITITION_TIME = 3 # Number of times repeated. 

ACOUSTIC_DELAY_DURATION = 0.0008 # Unit: s. 

# MOVING_INTERVAL_WIDTH = 100 # 50 for a break. 
PHOTO_SYNC_THRSLD = (0.2, 0.05) # Default: (0.2, 0.05). Minimum can be (0.1, 0.05) for tighter partitioning. 

# Define acoustic data partition & sampling strategies. 
AUDIO_FILE_EXTENSION = "npy" # Can be "wav" or "lvm". 
AUDIO_SAMPLING_RATE = 100e3 # Default: 96e3. Unit: Hz. Change according to the sensor's specs. 
AUDIO_CLIP_LENGTH_DP = 200 # Default: 160. 
AUDIO_CLIP_STRIDE_DP = 100 # Default: 80. 
SPECTRAL_TRANSFORM_KEYWORD = "cwt"

# Generate stft spectrogram (scalogram). 
N_FFT = 128
HOP_LENGTH = 64

# Generate wavelet spectrogram (scalogram). 
WAVELET = 'morl' # Default: Morlet ('morl'). 
SCALES = np.arange(2, 16, 0.5) # 1D Array of Int. The scale of the wavelet. 
SPECTRUM_DPI = 1200 # Default: 256. Intend to generate an image with size 256x256. 

# Spectrogram/Scalogram y-axis scale. 
IS_LOG_SCALE = False
IS_SAVE_CLIPS = True
IS_SAVE_SPECTRUMS = False # Current: Use Matlab to generate wavelet spectrums of clips. 
# IS_VISUALIZE = False
CLIP_FILE_EXTENSION = "mat" # Can be "npy" or "mat". 
SPECTRUM_FIG_EXTENSION = "png"
SPECTRUM_IMG_SIZE = 256


# Old dataset need the following param. 
IS_OMIT_DURATION = True # Used with `OMIT_DURATION`. 
OMIT_DURATION = [0.0720, 0.0619, 0.0638, 0.0682, 0.0658, 0.0696, 0.0686, 0.0731, 
                 0.0680, 0.0686, 0.0658, 0.0672, 0.0704, 0.0673, 0.0622, 0.0673, 
                 0.0657, 0.0717, 0.0628, 0.0622, 0.0696, 0.0669, 0.0660, 0.0680, 
                 0.0627, 0.0631, 0.0645, 0.0726, 0.0720] # Only used with the old data. 



# If high-speed sampling frequency is 22000 Hz, considering both the cumulative error and effective detection/featurization temporal resolution, the recommended time window length is 2 ms minimum (length - 200 dp / 44 imgs, stride - 100 dp / 22 imgs). 