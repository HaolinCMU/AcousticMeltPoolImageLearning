# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 19:25:52 2022

@author: hlinl
"""

import os 
import numpy as np
import matplotlib.pyplot as plt


def synchronize(audio_sample, photodiode_sample, sync_threshold=0.2,
                sr=100e3, acoustic_delay=0.0008):
    """
    Synchronize acoustic data with high-speed image data using photodiode.
    Cut off both the beginning and the end.  
    """
    
    acoustic_delay_inDP = int(sr*acoustic_delay)
    
    impinging_pt_begin = np.where(photodiode_sample>=sync_threshold)[0][0]
    impinging_pt_end = np.where(photodiode_sample>=sync_threshold)[0][-1]
    
    return audio_sample[impinging_pt_begin+acoustic_delay_inDP:impinging_pt_end+acoustic_delay_inDP], \
           photodiode_sample[impinging_pt_begin:impinging_pt_end]


acoustic_directory = "F:/data/raw/audio"
photo_directory = "F:/data/raw/photodiode"
audio_file_list = os.listdir(acoustic_directory)
photo_file_list = os.listdir(photo_directory)

length_list = []
for file in photo_file_list:
    acoustic_file_path = os.path.join(acoustic_directory, file)
    photo_file_path = os.path.join(photo_directory, file)
    acoustic_data = np.load(acoustic_file_path)[0,:]
    photo_data = np.load(photo_file_path)[0,:]
    
    acoustic_data_synced, _ = synchronize(acoustic_data, photo_data)
    length_list.append(len(acoustic_data_synced))
    
    del acoustic_data, photo_data, acoustic_data_synced
    # break

plt.figure()
plt.plot(length_list)