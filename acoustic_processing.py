# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 16:12:33 2022

@author: hlinl
"""


import os
import gc
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
                 clip_length=ACOUSTIC.AUDIO_CLIP_LENGTH_DP, clip_stride=ACOUSTIC.AUDIO_CLIP_STRIDE_DP):
                #  is_save_offline=ACOUSTIC.IS_SAVE, clips_subdir=BASIC.ACOUSTIC_CLIPS_FOLDER_PATH,
                #  spectrograms_subdir=BASIC.ACOUSTIC_SPECS_FOLDER_PATH):
        """
        """

        super(Clips, self).__init__(file_path, data, sr, omit)
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        # self.is_save_offline = is_save_offline
        # self.clips_subdir = clips_subdir
        # self.spectrograms_subdir = spectrograms_subdir

        self._audio_file_name = None
        self._index_list = None
        self._clips_mat = None
        self._clips_num = 0
        self._spectrogram_list = []

        self._get_audio_name()
        self.partition()
        self.spectral_transform()
    

    @property
    def clips(self):
        """
        """
        
        if self._clips_mat is not None: return self._clips_mat
        else: raise ValueError("Clips not partitioned. ")
    
    
    @property
    def spectrograms(self):
        """
        """
        
        if self._spectrogram_list != []: return self._spectrogram_list
        else: raise ValueError("Clips not spectrally transformed. Spectrograms not generated. ")


    def _get_audio_name(self):
        """
        """

        self._audio_file_name = Path(self.file_path).stem
    

    def partition(self):
        """
        """

        self._clips_mat = sliding_window_view(self.data, self.clip_length)[::self.clip_stride,:]
        self._clips_num = self._clips_mat.shape[0]
        self._index_list = [i+1 for i in range(self._clips_num)]
    

    def spectral_transform(self, spectrum_mode=ACOUSTIC.SPECTRAL_TRANSFORM_KEYWORD):
        """
        """

        self._spectrogram_list = []

        if self._clips_mat is None or self._clips_num != self._clips_mat.shape[0]: pass
        else:
            for ind in range(self._clips_num):
                if spectrum_mode == 'stft':
                    stft_transformer = audioBasics.STFTSpectrum(data=self._clips_mat[ind,:], sr=self.sampling_rate, 
                                                                n_fft=ACOUSTIC.N_FFT, hop_length=ACOUSTIC.HOP_LENGTH, 
                                                                is_log_scale=ACOUSTIC.IS_LOG_SCALE)
                    self._spectrogram_list.append(copy.deepcopy(stft_transformer.spectrum)) # List of 2D numpy array of STFT spectrogram images. 

                    del stft_transformer
                    gc.collect()

                elif spectrum_mode == 'cwt':
                    wavelet_transformer = audioBasics.WaveletSpectrum(data=self._clips_mat[ind,:], wavelet=ACOUSTIC.WAVELET, 
                                                                      scales=ACOUSTIC.SCALES, is_log_scale=ACOUSTIC.IS_LOG_SCALE)
                    self._spectrogram_list.append(copy.deepcopy(wavelet_transformer.spectrum)) # List of 2D numpy array of Wavelet spectrogram images. 

                    del wavelet_transformer
                    gc.collect()

                else: pass
    

    def _save_clips(self, clips_subdir):
        """
        """

        save_folder_path = os.path.join(clips_subdir, self._audio_file_name)

        if self._clips_mat is not None and self._index_list is not None:
            if not os.path.isdir(save_folder_path): os.mkdir(save_folder_path)

            for ind in range(self._clips_num):
                save_clip_path = os.path.join(save_folder_path, "{}.{}".format(str(self._index_list[ind]).zfill(5), 
                                                                               ACOUSTIC.CLIP_FILE_EXTENSION))
                np.save(save_clip_path, copy.deepcopy(self._clips_mat[ind,:].reshape(-1)))

        else: raise ValueError("Clips not partitioned. ")

    
    def _save_spectrograms(self, spectrograms_subdir):
        """
        """

        save_folder_path = os.path.join(spectrograms_subdir, self._audio_file_name)

        if self._spectrogram_list != [] and self._index_list is not None:
            if not os.path.isdir(save_folder_path): os.mkdir(save_folder_path)

            for ind, spectrogram in enumerate(self._spectrogram_list):
                save_spec_path = os.path.join(save_folder_path, "{}.{}".format(str(self._index_list[ind]).zfill(5), 
                                                                               ACOUSTIC.SPECTRUM_FIG_EXTENSION))
                plt.imsave(save_spec_path, spectrogram, dpi=ACOUSTIC.SPECTRUM_DPI)

        else: raise ValueError("Clips not spectrally transformed. Spectrograms not generated. ")
    
    
    def save_data_offline(self, clips_subdir=BASIC.ACOUSTIC_CLIPS_FOLDER_PATH, 
                          spectrograms_subdir=BASIC.ACOUSTIC_SPECS_FOLDER_PATH):
        """
        """

        self._save_clips(clips_subdir)
        self._save_spectrograms(spectrograms_subdir)

    
if __name__ == "__main__":
    """
    """

    
    
    clips_obj = Clips(file_path="data/raw_audio_data/Layer003_Section_01_S0001.wav", omit=0.0720)
    if ACOUSTIC.IS_SAVE: clips_obj.save_data_offline()
    
    del clips_obj
    gc.collect()
    
    # plt.figure(figsize=(20, 20))
    # plt.imshow(spectrograms[0])
    # plt.savefig('test.png', bbox_inches='tight', pad_inches=0)
    
