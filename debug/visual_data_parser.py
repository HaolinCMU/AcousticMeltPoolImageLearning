# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 02:00:02 2022

@author: hlinl
"""

import os
import glob
import copy
import sys
DIR_ABS = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR_ABS))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mig
import PIL
import re

from cmath import nan
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import skew, kurtosis
from scipy.ndimage import rotate
from torchvision import transforms

from image_processing import *
from PARAM.ACOUSTIC import *


img_dir = "F:/data/test"
visual_dir = "F:/data/processed/visual"
img_window_size = 36
img_stride = 18


img_subfolder_list = os.listdir(img_dir)

for folder in img_subfolder_list:
    area_list, aspect_ratio_list = copy.deepcopy([]), copy.deepcopy([])

    img_subfolder_dir = os.path.join(img_dir, folder)
    img_path_list_temp = glob.glob(os.path.join(img_subfolder_dir, "*.{}".format('png')))

    P_temp, V_temp = extract_process_param_fromAcousticFilename(folder)

    for ind, img_path in enumerate(img_path_list_temp):
        img_frame_temp = Frame(img_path)
        area_list.append(img_frame_temp.meltpool_area)
        aspect_ratio_list.append(img_frame_temp.meltpool_aspect_ratio)

        if (ind + 1) % 1000 == 0 or ind + 1 == len(img_path_list_temp): 
            print("Img: {} | {} is being processed. ".format(folder, ind+1))

        del img_frame_temp # Release memory. 
    
    area_array = np.array(area_list).astype(float).reshape(-1,1)
    aspect_ratio_array = np.array(aspect_ratio_list).astype(float).reshape(-1,1)
    P_array = np.array([P_temp]*len(img_path_list_temp)).astype(float).reshape(-1,1)
    V_array = np.array([V_temp]*len(img_path_list_temp)).astype(float).reshape(-1,1)

    visual_data_mat_full = np.hstack((area_array, aspect_ratio_array, P_array, V_array))
    visual_data_block = sliding_window_view(visual_data_mat_full, 
                                            window_shape=(img_window_size, 
                                                          visual_data_mat_full.shape[1]))[::img_stride,:,:,:] # Shape: (IMAGE_WINDOW_SIZE, feature_num, sample_num). 
    visual_data_mat = np.mean(visual_data_block, axis=2).reshape(-1, visual_data_mat_full.shape[1])

    visual_subfolder_dir = os.path.join(visual_dir, folder)
    if not os.path.isdir(visual_subfolder_dir): os.mkdir(visual_subfolder_dir)

    for i in range(visual_data_mat.shape[0]):
        visual_data_path_temp = os.path.join(visual_subfolder_dir, "{}_{}.npy".format(folder, str(i).zfill(5)))
        np.save(visual_data_path_temp, visual_data_mat[i,:])

