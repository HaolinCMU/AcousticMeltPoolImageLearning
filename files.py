# -*- coding: utf-8 -*-
"""
Created on Sat May 21 03:26:25 2022

@author: hlinl
"""


import os
import glob
import copy
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mig

from PARAM import *
from dataprocessing import *


def clr_type(directory, file_extension):
    """
    """
    
    file_path_list = glob.glob(os.path.join(directory, "*.{}".format(file_extension)))
    for file_path in file_path_list: os.remove(file_path)


def clr_dir(directory):
    """
    """
    
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isdir(path): 
            shutil.rmtree(path)
        else: os.remove(path)

    
# def rename(src, dst):
#     """
#     """

#     os.rename(src, dst)


if __name__ == "__main__":
    """
    """

    directory = "data/raw_image_data"
    img_subfolder_list = os.listdir(directory) # List of subfolders of different layers.

    for img_subfolder in img_subfolder_list:
        img_subfolder_path = os.path.join(directory, img_subfolder)
        img_filepath_perSubFolder_list = glob.glob(os.path.join(img_subfolder_path, 
                                                                "*.{}".format('cihx'))) # List of image file paths of each layer's subfolder. 

        for _, img_filepath in enumerate(img_filepath_perSubFolder_list):
            os.remove(img_filepath)