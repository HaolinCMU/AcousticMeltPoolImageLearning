# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:14:22 2022

@author: hlinl
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils


# Define acoustic data partition & sampling strategies. 
AUDIO_FILE_EXTENSION = "wav" # Can be "wav" or "lvm". 
CLIP_FILE_EXTENSION = "npy"
AUDIO_SAMPLING_RATE = 96e3 # Default: 96e3. Unit: Hz. Change according to the sensor's specs. 
AUDIO_CLIP_LENGTH_DP = 128 # Default: 128. 
AUDIO_CLIP_STRIDE_DP = 64 # Default: 64. 
IS_OMIT_DURATION = True # Used with `OMIT_DURATION`. 
OMIT_DURATION = [0.0720, 0.0619, 0.0638, 0.0682, 0.0658, 0.0696, 0.0686, 0.0731, 
                 0.0680, 0.0686, 0.0658, 0.0672, 0.0704, 0.0673, 0.0622, 0.0673, 
                 0.0657, 0.0717, 0.0628, 0.0622, 0.0696, 0.0669, 0.0660, 0.0680, 
                 0.0627, 0.0631, 0.0645, 0.0726, 0.0720] # Only used with the old data. 

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
IS_SAVE = True
# IS_VISUALIZE = False
SPECTRUM_FIG_FOLDER = "data/acoustic_data"
SPECTRUM_FIG_EXTENSION = "png"





