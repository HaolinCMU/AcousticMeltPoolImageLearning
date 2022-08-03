# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:36:13 2022

@author: hlinl
"""


import os
import numpy as np


# DATA INFORMATION
DATA_DIRECTORY = "data"

# IMAGE PATHS. 
IMAGE_DATA_SUBDIR = "raw_image_data"
IMAGE_PROCESSED_DATA_SUBDIR = "img_data"
IMAGE_EXTENSION = "png"

# ACOUSTIC PATHS. 
AUDIO_DATA_SUBDIR = "raw_audio_data"
ACOUSTIC_PROCESSED_DATA_SUBDIR = "acoustic_data"
ACOUSTIC_CLIPS_SUBSUBFOLDER = "acoustic_clips_data"
ACOUSTIC_SPECS_SUBSUBFOLDER = "acoustic_spectrums_data"
ACOUSTIC_CLIPS_FOLDER_PATH = os.path.join(DATA_DIRECTORY, ACOUSTIC_PROCESSED_DATA_SUBDIR, 
                                          ACOUSTIC_CLIPS_SUBSUBFOLDER)
ACOUSTIC_SPECS_FOLDER_PATH = os.path.join(DATA_DIRECTORY, ACOUSTIC_PROCESSED_DATA_SUBDIR, 
                                          ACOUSTIC_SPECS_SUBSUBFOLDER)
ACOUSTIC_EXTENSION = "wav" # Acoustic data is saved into 'npy' format. 'wav', 'lvm' or 'npy'. 


HU_MOMENTS_FEATURE_IND_LIST = [0,1,2,3,4,5,6] # Must be sorted. 




