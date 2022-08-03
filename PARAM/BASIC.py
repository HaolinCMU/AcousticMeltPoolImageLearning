# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:36:13 2022

@author: hlinl
"""


import os
import numpy as np


# DATA INFORMATION

DATA_DIRECTORY = "data"
IMAGE_DATA_SUBDIR = "raw_image_data"
AUDIO_DATA_SUBDIR = "raw_audio_data"
IMAGE_PROCESSED_DATA_SUBDIR = "img_data"
ACOUSTIC_PROCESSED_DATA_SUBDIR = "acoustic_data"

IMAGE_EXTENSION = "png"
HU_MOMENTS_FEATURE_IND_LIST = [0,1,2,3,4,5,6] # Must be sorted. 

ACOUSTIC_EXTENSION = "wav" # Acoustic data is saved into 'npy' format. 'wav', 'lvm' or 'npy'. 


