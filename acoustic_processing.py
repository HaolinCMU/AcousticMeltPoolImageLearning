# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 16:12:33 2022

@author: hlinl
"""


import os
import glob
import copy

import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mig
import PIL

from scipy.stats import skew, kurtosis
from scipy.ndimage import rotate
from torchvision import transforms

from PARAM import *
from dataprocessing import *
from files import *


class Clips(audioBasics.Audio):
    """
    """

    def __init__(self, file_path=None, data=None, sr=ACOUSTIC.AUDIO_SAMPLING_RATE, omit=0., 
                 clip_length=ACOUSTIC.AUDIO_CLIP_LENGTH_DP, clip_stride=ACOUSTIC.AUDIO_CLIP_STRIDE_DP):
        """
        """

        super(Clips, self).__init__(file_path, data, sr, omit)
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        
        self._index_list = None
        self._clip_list = None
        self._spectrogram_list = None

        self.partition() # Partition the input data into short clips as per given param. 
    
    
    @property
    def clips(self):
        """
        """
        
        return self._clip_list
    
    
    @property
    def spectrograms(self):
        """
        """
        
        return self._spectrogram_list
    
    
    def partition(self):
        """
        """

        self._index_list, self._clip_list = [], []
        clip_num = (self.audio_len - self.clip_stride) // self.clip_stride + 1

        for i in range(clip_num):
            start = i*self.clip_stride
            end = start + self.clip_length

            clip_temp = copy.deepcopy(self.data[start:end])
            self._clip_list.append(clip_temp)
            self._index_list.append(i+1)

    
    def spectral_transform(self, transform_keyword=ACOUSTIC.SPECTRAL_TRANSFORM_KEYWORD):
        """
        """

        self._spectrogram_list = []

        if self._clip_list is None: pass
        else:
            for ind, clip in enumerate(self._clip_list):
                if ACOUSTIC.IS_SAVE: 
                    save_fig_path = os.path.join(ACOUSTIC.SPECTRUM_FIG_FOLDER, 
                                                 "{}.{}".format(self._index_list[ind], ACOUSTIC.SPECTRUM_FIG_EXTENSION))
                
                if transform_keyword == 'stft':
                    stft_transformer = audioBasics.STFTSpectrum(clip, self.sampling_rate, ACOUSTIC.N_FFT, 
                                                                ACOUSTIC.HOP_LENGTH, ACOUSTIC.IS_LOG_SCALE)
                    self._spectrogram_list.append(stft_transformer.spectrogram) # List of 2D numpy array of STFT spectrogram images. 

                elif transform_keyword == 'cwt':
                    wavelet_transformer = audioBasics.WaveletSpectrum(clip, ACOUSTIC.WAVELET, ACOUSTIC.SCALE, 
                                                                      ACOUSTIC.IS_LOG_SCALE, ACOUSTIC.IS_SAVE, 
                                                                      ACOUSTIC.IS_VISUALIZE, save_fig_path)
                    self._spectrogram_list.append(wavelet_transformer.spectrum) # List of 2D numpy array of Wavelet spectrogram images. 

                else: pass

    
if __name__ == "__main__":
    """
    """
    
    clips_obj = Clips(file_path="data/raw_audio_data/Layer003_Section_01_S0001.wav", omit=0.0720)
    clips_obj.spectral_transform()
    spectrograms = clips_obj.spectrograms
    
