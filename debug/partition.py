# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:50:31 2022

@author: hlinl
"""


import numpy as np
import scipy.io

from numpy.lib.stride_tricks import sliding_window_view


def synchronize(audio_sample, photodiode_sample, sync_threshold=0.05):
    """
    Synchronize acoustic data with high-speed image data using photodiode.
    Cut off both the beginning and the end.  
    """
    
    impinging_pt_begin = np.where(photodiode_sample>=sync_threshold)[0][0]
    impinging_pt_end = np.where(photodiode_sample>=sync_threshold)[0][-1]

    return audio_sample[impinging_pt_begin:impinging_pt_end], \
           photodiode_sample[impinging_pt_begin:impinging_pt_end]


if __name__ == "__main__":
    """
    """
    
    file_name = 'Layer0021_P200_V0250_C001H001S0001'

    audio_dir = 'F:/data/raw/audio/{}.npy'.format(file_name)
    clips_mat_dir = 'F:/data/processed/acoustic/clips/{}/{}.mat'.format(file_name, file_name)
    photodiode_dir = 'F:/data/raw/photodiode/{}.npy'.format(file_name)
    
    clip_length = 160
    clip_stride = 80
    clip_No = 36 # Start from 1. 
    
    photo_thrsld = 0.05
    audio_sensor = 0
    photo_sensor = 0
    
    audio_sample = np.load(audio_dir)[audio_sensor,:]
    photo_sample = np.load(photodiode_dir)[photo_sensor,:]
    clips_mat = scipy.io.loadmat(clips_mat_dir)['clips_mat']
    
    audio, photo = synchronize(audio_sample, photo_sample, sync_threshold=photo_thrsld)
    
    start_ind = (clip_No-1)*clip_stride
    segment = audio[start_ind:start_ind+clip_length]
    clip = clips_mat[clip_No-1,:]
    
    print((segment==clip).all())
    print(clips_mat.shape[0])
    


