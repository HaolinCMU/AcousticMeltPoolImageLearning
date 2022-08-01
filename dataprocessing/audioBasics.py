# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 16:46:26 2022

@author: hlinl
"""


import os
import copy
import sys
DIR_ABS = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR_ABS))

import librosa
import matplotlib.image as mig
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy

from PARAM import *

      
class STFTSpectrum(object):
    """
    """

    def __init__(self, data, sr, n_fft, hop_length, is_log_scale=False):
        """
        """

        self.acoustic_data = data
        self.sampling_rate = sr

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.is_log_scale = is_log_scale
        self.y_axis = 'log' if self.is_log_scale else 'linear'

        self._coef = None
        self._spectrogram = None

        self.transform()
    

    @property
    def coef(self):
        """
        """

        return self._coef

    
    @property
    def spectrogram(self):
        """
        """

        return self._spectrogram

    
    def transform(self):
        """
        """

        self._coef = np.abs(librosa.stft(self.acoustic_data, n_fft=self.n_fft, hop_length=self.hop_length))
        self._spectrogram = librosa.amplitude_to_db(self._coef, ref=np.max)

    
    def plot(self):
        """
        """

        fig, ax = plt.figure()
        img = librosa.display.specshow(self._spectrogram, y_axis=self.y_axis, sr=self.sampling_rate, 
                                       hop_length=self.hop_length, n_fft=self.n_fft, x_axis='time')
        fig.colorbar(img, ax=ax)
        fig.canvas.draw()

        self._spectrogram = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).\
                                        reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # plt.show()


class WaveletSpectrum(object):
    """
    """

    def __init__(self, data, wavelet, scales, map_resolution, is_log_scale=True):
        """
        """

        self.acoustic_data = data
        self.wavelet = wavelet
        self.scales = scales
        self.map_resolution = map_resolution
        self.is_log_scale = is_log_scale

        self.coef = None
        self.freqs = None
        self._spectrum = None
    

    @property
    def coef(self):
        """
        """

        return self.coef

    
    @property
    def freqs(self):
        """
        """

        return self.freqs

    
    @property
    def spectrum(self):
        """
        """

        return self._spectrum

    
    def transform(self):
        """
        """

        self.coef, self.freqs = pywt.cwt(self.acoustic_data, self.scales, self.wavelet)

    
    def plot(self):
        """
        """

        fig, ax = plt.figure()
        
        

        # self._spectrum = image matrix. 

        pass


class Audio(object):
    """
    """

    def __init__(self, file_path=None, data=None, sr=ACOUSTIC.AUDIO_SAMPLING_RATE, omit=0.):
        """
        """

        self.file_path = file_path
        self.sampling_rate = sr
        self.omit_duration = omit # Unit: In s.

        self.data = data if data is not None and self.file_path is None else \
                    self._read_data_from_path(self.file_path)
        self.audio_len = len(self.data)
    

    def __len__(self):
        """
        """

        return self.audio_len


    def _read_data_from_path(self):
        """
        """

        if self.file_path is None: y = np.zeros(shape=(ACOUSTIC.AUDIO_CLIP_LENGTH_DP,))
        else: y, _ = librosa.load(self.file_path, sr=self.sampling_rate, offset=self.omit_duration)

        return y.reshape(-1)

    
    def stft(self, n_fft, hop_length):
        """
        """

        stft_spectral = STFTSpectrum(self.data, self.sampling_rate, n_fft, hop_length)
        return stft_spectral.spectrogram


    def waveplot(self, duration=None, start_ind=None, end_ind=None, fig_path=None):
        """
        Duration: Tuple of Floats. (start_time, end_time). In s. 
        Duration prioritizes `start_ind` and `end_ind`. 
        """

        if duration is not None: start_ind, end_ind = (np.array(duration)*self.sampling_rate).astype(int).reshape(-1)
        else: pass

        waveplot_label = "waveplot:{:.4f}-{:.4f}s".format(duration[0], duration[1])

        plt.figure()
        plt.plot(self.data[start_ind:end_ind], linewidth=3.0, label=waveplot_label)
        if fig_path is not None: plt.savefig(fig_path)
        plt.show()

