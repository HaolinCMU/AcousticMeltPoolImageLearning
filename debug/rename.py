# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 02:52:39 2022

@author: hlinl
"""


import os
import glob
import shutil


directory = "F:/data/raw/photodiode"
file_list = os.listdir(directory)

for file in file_list:
    old_path = os.path.join(directory, file)
    new_file = '_'.join(file.split('_')[2:])
    new_path = os.path.join(directory, new_file)
    os.rename(old_path, new_path)
