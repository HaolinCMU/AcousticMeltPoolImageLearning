# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:36:13 2022

@author: hlinl
"""


import os
import numpy as np


#### NEW DATA ####

# DATA INFORMATION
DATA_DIRECTORY = "F:/data" # Default: "data"

# IMAGE PATHS. 
IMAGE_DATA_SUBDIR = "F:/data/raw/highspeed" # "raw_image_data"
IMAGE_PROCESSED_DATA_SUBDIR = "F:/data/processed/image"

# ACOUSTIC PATHS. 
AUDIO_DATA_SUBDIR = "F:/data/raw/audio" # Default: "raw_audio_data". 
ACOUSTIC_PROCESSED_DATA_SUBFOLDER = "processed/acoustic"
ACOUSTIC_CLIPS_SUBSUBFOLDER = "clips"
ACOUSTIC_SPECS_SUBSUBFOLDER = "wavelet_spectrums_short"
ACOUSTIC_CLIPS_FOLDER_PATH = os.path.join(DATA_DIRECTORY, ACOUSTIC_PROCESSED_DATA_SUBFOLDER, 
                                          ACOUSTIC_CLIPS_SUBSUBFOLDER)
ACOUSTIC_SPECS_FOLDER_PATH = os.path.join(DATA_DIRECTORY, ACOUSTIC_PROCESSED_DATA_SUBFOLDER, 
                                          ACOUSTIC_SPECS_SUBSUBFOLDER)
ACOUSTIC_EXTENSION = "npy" # Acoustic data is saved into 'npy' format. 'wav', 'lvm' or 'npy'. 

PHOTODIODE_DATA_SUBDIR = "F:/data/raw/photodiode"


# VISUAL DATA PARAM.
VISUAL_DATA_SUBDIR = "F:/data/processed/visual"
HU_MOMENTS_FEATURE_IND_LIST = [0,1,2,3,4,5,6] # Must be sorted. 




#### OLD DATA ####

# # DATA INFORMATION
# DATA_DIRECTORY = "data"

# # IMAGE PATHS. 
# IMAGE_DATA_SUBDIR = "raw_image_data"
# IMAGE_PROCESSED_DATA_SUBDIR = "img_data"
# IMAGE_EXTENSION = "png"

# # ACOUSTIC PATHS. 
# AUDIO_DATA_SUBDIR = "raw_audio_data"
# ACOUSTIC_PROCESSED_DATA_SUBFOLER = "acoustic_data"
# ACOUSTIC_CLIPS_SUBSUBFOLDER = "acoustic_clips_data"
# ACOUSTIC_SPECS_SUBSUBFOLDER = "acoustic_spectrums_data"
# ACOUSTIC_CLIPS_FOLDER_PATH = os.path.join(DATA_DIRECTORY, ACOUSTIC_PROCESSED_DATA_SUBFOLER, 
#                                           ACOUSTIC_CLIPS_SUBSUBFOLDER)
# ACOUSTIC_SPECS_FOLDER_PATH = os.path.join(DATA_DIRECTORY, ACOUSTIC_PROCESSED_DATA_SUBFOLER, 
#                                           ACOUSTIC_SPECS_SUBSUBFOLDER)
# ACOUSTIC_EXTENSION = "wav" # Acoustic data is saved into 'npy' format. 'wav', 'lvm' or 'npy'. 

# # VISUAL DATA PATHS. 
# VISUAL_DATA_SUBFOLDER = "visual_data"
# VUSIAL_DATA_SUBDIR = os.path.join(DATA_DIRECTORY, VISUAL_DATA_SUBFOLDER)

# HU_MOMENTS_FEATURE_IND_LIST = [0,1,2,3,4,5,6] # Must be sorted. 