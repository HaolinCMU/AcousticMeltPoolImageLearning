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

from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

from PARAM import *
from dataprocessing import *
from files import *


class Clips(audioBasics.Audio):
    """
    """

    def __init__(self, file_path=None, data=None, sr=ACOUSTIC.AUDIO_SAMPLING_RATE, omit=0.,
                 clip_length=ACOUSTIC.AUDIO_CLIP_LENGTH_DP, clip_stride=ACOUSTIC.AUDIO_CLIP_STRIDE_DP,
                 is_save_offline=ACOUSTIC.IS_SAVE, clips_subdir=BASIC.ACOUSTIC_CLIPS_FOLDER_PATH,
                 spectrograms_subdir=BASIC.ACOUSTIC_SPECS_FOLDER_PATH):
        """
        """

        super(Clips, self).__init__(file_path, data, sr, omit)
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.is_save_offline = is_save_offline
        self.clips_subdir = clips_subdir
        self.spectrograms_subdir = spectrograms_subdir

        self._audio_file_name = None
        self._index_list = None
        self._clips_mat = None
        # self._spectrogram_list = None
    

    def _get_audio_name(self):
        """
        """

        self._audio_file_name = Path(self.file_path).stem






class Clips(audioBasics.Audio):
    """
    """

    def __init__(self, file_path=None, data=None, sr=ACOUSTIC.AUDIO_SAMPLING_RATE, omit=0., 
                 clip_length=ACOUSTIC.AUDIO_CLIP_LENGTH_DP, clip_stride=ACOUSTIC.AUDIO_CLIP_STRIDE_DP, 
                 is_save=ACOUSTIC.IS_SAVE, 
                 save_clips_folder_path=BASIC.ACOUSTIC_CLIPS_FOLDER_PATH, 
                 save_specs_folder_path=BASIC.ACOUSTIC_SPECS_FOLDER_PATH):
        """
        """

        super(Clips, self).__init__(file_path, data, sr, omit)
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.is_save = is_save
        self.save_clips_folder_path = save_clips_folder_path
        self.save_specs_folder_path = save_specs_folder_path
        
        self._audio_file_name = None # Name without extension. 
        self._index_list = None
        self._clip_list = None
        self._spectrogram_list = None

        self._get_audio_name() 
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
    
    
    def _get_audio_name(self):
        """
        """

        self._audio_file_name = Path(self.file_path).stem


    def partition(self):
        """
        """

        self._index_list, self._clip_list = [], []
        clip_num = (self.audio_len - self.clip_length) // self.clip_stride + 1

        for i in range(clip_num):
            start = i*self.clip_stride
            end = start + self.clip_length

            clip_temp = copy.deepcopy(self.data[start:end])
            self._clip_list.append(clip_temp)
            self._index_list.append(i+1)

        if self.is_save: self._save_clips()

        clips_mat = sliding_window_view(self.data, self.clip_length)[::self.clip_stride,:]

    
    def spectral_transform(self, transform_keyword=ACOUSTIC.SPECTRAL_TRANSFORM_KEYWORD):
        """
        """

        self._spectrogram_list = []

        if self._clip_list is None: pass
        else:
            for clip in self._clip_list:
                if transform_keyword == 'stft':
                    stft_transformer = audioBasics.STFTSpectrum(data=clip, sr=self.sampling_rate, 
                                                                n_fft=ACOUSTIC.N_FFT, hop_length=ACOUSTIC.HOP_LENGTH, 
                                                                is_log_scale=ACOUSTIC.IS_LOG_SCALE)
                    self._spectrogram_list.append(copy.deepcopy(stft_transformer.spectrum)) # List of 2D numpy array of STFT spectrogram images. 

                elif transform_keyword == 'cwt':
                    wavelet_transformer = audioBasics.WaveletSpectrum(data=clip, wavelet=ACOUSTIC.WAVELET, 
                                                                      scales=ACOUSTIC.SCALE, 
                                                                      is_log_scale=ACOUSTIC.IS_LOG_SCALE)
                    self._spectrogram_list.append(copy.deepcopy(wavelet_transformer.spectrum)) # List of 2D numpy array of Wavelet spectrogram images. 

                else: pass
            
            if self.is_save and self._spectrogram_list != []: self._save_spectrograms()
    
    
    def _save_clips(self):
        """
        """

        save_folder_path = os.path.join(self.save_clips_folder_path, self._audio_file_name)

        if self.is_save and \
           self._clip_list is not None and \
           self._index_list is not None:
            if not os.path.isdir(save_folder_path): os.mkdir(save_folder_path)

            for ind, clip in enumerate(self._clip_list):
                save_clip_path = os.path.join(save_folder_path, "{}.{}".format(str(self._index_list[ind]).zfill(5), 
                                                                               ACOUSTIC.CLIP_FILE_EXTENSION))
                np.save(save_clip_path, copy.deepcopy(clip.reshape(-1)))

        else: pass

    
    def _save_spectrograms(self):
        """
        """

        save_folder_path = os.path.join(self.save_specs_folder_path, self._audio_file_name)

        if self.is_save and \
           self._spectrogram_list is not None and \
           self._index_list is not None:
            if not os.path.isdir(save_folder_path): os.mkdir(save_folder_path)

            for ind, spectrogram in enumerate(self._spectrogram_list):
                save_spec_path = os.path.join(save_folder_path, "{}.{}".format(str(self._index_list[ind]).zfill(5), 
                                                                               ACOUSTIC.SPECTRUM_FIG_EXTENSION))
                plt.imsave(save_spec_path, spectrogram, dpi=ACOUSTIC.SPECTRUM_DPI)

        else: pass

    
if __name__ == "__main__":
    """
    """
    
    clips_obj = Clips(file_path="data/raw_audio_data/Layer003_Section_01_S0001.wav", omit=0.0720)
    clips_obj.spectral_transform()
    spectrograms = clips_obj.spectrograms
    
    # plt.figure(figsize=(20, 20))
    # plt.imshow(spectrograms[0])
    # plt.savefig('test.png', bbox_inches='tight', pad_inches=0)
    
