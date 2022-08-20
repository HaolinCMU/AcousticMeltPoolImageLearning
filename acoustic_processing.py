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
import scipy.io

from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

from PARAM import *
from dataprocessing import *
from files import *


class Synchronizer(object):
    """
    Use photodiode data to synchronize acoustic and high-speed image data. Cut off both the beginning and the end of the acoustic data.  
    """

    def __init__(self, acoustic_data_file_path=None, acoustic_data=None, photodiode_data_file_path=None, 
                 photodiode_data=None, data_label=None, sr=ACOUSTIC.AUDIO_SAMPLING_RATE, 
                 photodiode_sync_thrsld=ACOUSTIC.PHOTO_SYNC_THRSLD, 
                 acoustic_delay_duration=ACOUSTIC.ACOUSTIC_DELAY_DURATION):
        """
        Acoustic data must have the same sampling rate as the corresponding photodiode data. 
        `acoustic_delay_duration`: Since acoustic data is synced with high-speed by photodiode data, there will be a short delay in the collected acoustic signal, depending on the location of the acoustic sensor in the machine. Unit: s. 
        """

        self._audio_sensor_No = ACOUSTIC.AUDIO_SENSOR_NO # Int. Indicate which aocustic sensor is chosen. 
        self._photodiode_sensor_No = ACOUSTIC.PHOTODIODE_SENSOR_NO # Int. Indicate which photodiode sensor is chosen. 
        self._photodiode_sync_thrsld = photodiode_sync_thrsld # Tuple of Float. The head and end magnitude threshold of photodiode data. 

        self.acoustic_data_filepath = acoustic_data_file_path
        self.photodiode_data_filepath = photodiode_data_file_path
        self.acoustic_data = acoustic_data if acoustic_data is not None else self._read_data_from_path()[0]
        self.photodiode_data = photodiode_data if photodiode_data is not None else self._read_data_from_path()[1]
        self.audio_sr = sr # Sampling frequency of acoustic signals.
        self.acoustic_delay_duration = acoustic_delay_duration

        self._audio_file_name = data_label if data_label is not None else self._get_audio_name()
        self._acoustic_data_synced_dict = defaultdict()

        self.synchronize()
    

    def _get_audio_name(self):
        """
        """

        if self.acoustic_data_filepath is not None: self.audio_file_name = Path(self.acoustic_data_filepath).stem
        else: self.audio_file_name = "audio_0" # Default name of audio files' folder. 


    def _read_data_from_path(self):
        """
        """

        if self.acoustic_data_filepath is None and self.photodiode_data_filepath is None: 
            raise ValueError("Audio data and phodiode data not found. ")

        acoustic_data, photodiode_data = None, None

        if self.acoustic_data_filepath is not None: 
            acoustic_file_extension = os.path.splitext(self.acoustic_data_filepath)[1]
            if acoustic_file_extension == ".wav": 
                acoustic_data, _ = librosa.load(self.acoustic_data_filepath, sr=self.audio_sr, offset=0.)
            elif acoustic_file_extension == ".npy": 
                acoustic_data = np.load(self.acoustic_data_filepath)
            elif acoustic_file_extension == ".csv":
                acoustic_data = np.genfromtxt(self.acoustic_data_filepath, delimiter=',')
            else: raise ValueError("Audio file format not recognizable. ")
        else: pass
        
        if self.photodiode_data_filepath is not None:
            photodiode_file_extension = os.path.splitext(self.photodiode_data_filepath)[1]
            if photodiode_file_extension == ".npy": 
                photodiode_data = np.load(self.photodiode_data_filepath)
            elif photodiode_file_extension == ".csv":
                photodiode_data = np.genfromtxt(self.photodiode_data_filepath, delimiter=',')
            else: raise ValueError("Photodiode file format not recognizable. ")
        else: pass

        return acoustic_data, photodiode_data


    def synchronize(self, photo_sensor_No=None):
        """
        """

        if photo_sensor_No is None: photo_sensor_No = self._photodiode_sensor_No

        photodiode_sample = self.photodiode_data[photo_sensor_No,:]
        audio_sensor_num = self.acoustic_data.shape[0]

        for i in range(audio_sensor_num):
            acoustic_sample = copy.deepcopy(self.acoustic_data[i,:])
            acoustic_sample_synced, _ = audioBasics.synchronize(acoustic_sample, photodiode_sample, sr=self.audio_sr, 
                                                                sync_threshold=self._photodiode_sync_thrsld,
                                                                acoustic_delay_duration=self.acoustic_delay_duration)
            self._acoustic_data_synced_dict[i] = acoustic_sample_synced


    def acoustic_synced_data(self, audio_sensor_No=None):
        """
        """

        if len(self._acoustic_data_synced_dict) != 0: 
            if audio_sensor_No is None: audio_sensor_No = self._audio_sensor_No
            return self._acoustic_data_synced_dict[audio_sensor_No]
        else: raise ValueError("Synchronization not implemented. ")


class Clips(audioBasics.Audio):
    """
    """

    def __init__(self, file_path=None, data=None, data_label=None, sr=ACOUSTIC.AUDIO_SAMPLING_RATE, omit=0.,
                 clip_length=ACOUSTIC.AUDIO_CLIP_LENGTH_DP, clip_stride=ACOUSTIC.AUDIO_CLIP_STRIDE_DP):
        """
        """

        super(Clips, self).__init__(file_path, data, data_label, sr, omit)
        self.clip_length = clip_length
        self.clip_stride = clip_stride

        self._index_list = None
        self._clips_mat = None
        self._clips_num = 0
        self._spectrogram_list = []

        self.partition()
    

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

        if self._spectrogram_list == []: self.spectral_transform()
        
        if self._spectrogram_list != []: return self._spectrogram_list
        else: raise ValueError("Clips not transformed to spectrums. Spectrograms not generated. ")


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

        save_folder_path = os.path.join(clips_subdir, self.audio_file_name)

        if self._clips_mat is not None and self._index_list is not None:
            if not os.path.isdir(save_folder_path): os.mkdir(save_folder_path)

            if ACOUSTIC.CLIP_FILE_EXTENSION == "mat":
                mdict = {"clips_mat": self._clips_mat}
                scipy.io.savemat(os.path.join(save_folder_path, "{}.mat".format(self.audio_file_name)), mdict)
            else:
                for ind in range(self._clips_num):
                    save_clip_path = os.path.join(save_folder_path, "{}.{}".format(str(self._index_list[ind]).zfill(5), 
                                                                                   ACOUSTIC.CLIP_FILE_EXTENSION))

                    if ACOUSTIC.CLIP_FILE_EXTENSION == "npy":
                        np.save(save_clip_path, copy.deepcopy(self._clips_mat[ind,:].reshape(-1)))
                    else: raise ValueError("Clip file extension not well defined. ")

        else: raise ValueError("Clips not partitioned. ")

    
    def _save_spectrograms(self, spectrograms_subdir):
        """
        """

        save_folder_path = os.path.join(spectrograms_subdir, self.audio_file_name)

        if self._spectrogram_list != [] and self._index_list is not None:
            if not os.path.isdir(save_folder_path): os.mkdir(save_folder_path)

            for ind, spectrogram in enumerate(self._spectrogram_list):
                save_spec_path = os.path.join(save_folder_path, "{}.{}".format(str(self._index_list[ind]).zfill(5), 
                                                                               ACOUSTIC.SPECTRUM_FIG_EXTENSION))
                
                if ACOUSTIC.SPECTRUM_FIG_EXTENSION == "png" or ACOUSTIC.SPECTRUM_FIG_EXTENSION == "jpg":
                    plt.imsave(save_spec_path, spectrogram, dpi=ACOUSTIC.SPECTRUM_DPI)
                else: raise ValueError("Spectrogram file extension not well defined. ")

        else: raise ValueError("Clips not spectrally transformed. Spectrograms not generated. ")
    
    
    def save_data_offline(self, is_clips=False, clips_subdir=BASIC.ACOUSTIC_CLIPS_FOLDER_PATH, 
                          is_spectrums=False, spectrograms_subdir=BASIC.ACOUSTIC_SPECS_FOLDER_PATH):
        """
        """

        if is_clips: self._save_clips(clips_subdir)
        if is_spectrums: 
            self.spectral_transform()
            self._save_spectrograms(spectrograms_subdir)

    
if __name__ == "__main__":
    """
    """

    pass
