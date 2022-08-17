# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 22:19:41 2022

@author: hlinl
"""

import os
import glob
import shutil


high_speed_dir = "F:\with_powder\TI64_acousticmeltpool_7.29.22_28rem\highspeed"
dst_dir = "F:\data\highspeed"
highspeed_img_folder_list = os.listdir(high_speed_dir)

for folder in highspeed_img_folder_list:
    src_subfolder_path = os.path.join(high_speed_dir, folder, "Images_S0001")
    dst_subfolder_path = os.path.join(dst_dir, folder)
    # if not os.path.isdir(dst_subfolder_path): os.mkdir(dst_subfolder_path)
    
    shutil.copytree(src_subfolder_path, dst_subfolder_path)


