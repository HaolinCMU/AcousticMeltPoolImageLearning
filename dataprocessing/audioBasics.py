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

from pathlib import Path

import librosa
import matplotlib.image as mig
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy

from PARAM import *


def synchronize(audio_sample, photodiode_sample, sr, sync_threshold=(0.2, 0.05), acoustic_delay_duration=0.):
    """
    Synchronize acoustic data with high-speed image data using photodiode.
    Cut off both the beginning and the end. 
    `acoustic_delay_duration`: Since acoustic data is synced with high-speed by photodiode data, there will be a short delay in the collected acoustic signal, depending on the location of the acoustic sensor in the machine. Unit: s. 
    """

    acoustic_delay_inDP = int(sr*acoustic_delay_duration)
    
    impinging_pt_begin = np.where(photodiode_sample>=sync_threshold[0])[0][0]
    impinging_pt_end = np.where(photodiode_sample>=sync_threshold[1])[0][-1]

    return audio_sample[impinging_pt_begin+acoustic_delay_inDP:impinging_pt_end+acoustic_delay_inDP], \
           photodiode_sample[impinging_pt_begin:impinging_pt_end]


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
        # self._spectrogram = librosa.amplitude_to_db(self._coef, ref=np.max)
        self.plot(visualize=False)

    
    def plot(self, visualize=True):
        """
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        img = librosa.display.specshow(self._spectrogram, y_axis=self.y_axis, sr=self.sampling_rate, 
                                       hop_length=self.hop_length, n_fft=self.n_fft, x_axis='time')
        # fig.colorbar(img, ax=ax)
        fig.canvas.draw()
        self._spectrogram = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).\
                                          reshape(fig.canvas.get_width_height()[::-1] + (3,)) # Save the plot from buffer as a numpy array. 
        
        if visualize: plt.show()

        plt.close()
        plt.cla()
        plt.clf()


class WaveletSpectrum(object):
    """
    """

    def __init__(self, data, wavelet, scales, is_log_scale=ACOUSTIC.IS_LOG_SCALE, 
                 is_save=False, is_visualize=False, spectrum_fig_path=None):
        """
        """

        self.acoustic_data = data
        self.wavelet = wavelet
        self.scales = scales
        self.is_log_scale = is_log_scale
        self.is_save = is_save
        self.is_visualize = is_visualize
        self.spectrum_fig_path = spectrum_fig_path

        self._coef = None
        self._freqs = None
        self._spectrum = None
        
        self.transform()
    

    @property
    def coef(self):
        """
        """

        return self._coef

    
    @property
    def freqs(self):
        """
        """

        return self._freqs

    
    @property
    def spectrum(self):
        """
        """

        return self._spectrum

    
    def transform(self):
        """
        """

        self._coef, self._freqs = pywt.cwt(self.acoustic_data, self.scales, self.wavelet)
        self.plot(self.is_save, self.is_visualize)

    
    def plot(self, save=False, visualize=True):
        """
        """

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_axes([0.,0.,1.,1.])
        extent = [-1, 1, 1, len(self.scales)+1]
        _ = ax.imshow(abs(self._coef), extent=extent, interpolation='bilinear', cmap='gray', aspect='auto',
                      vmax=abs(self._coef).max(), vmin=-abs(self._coef).max())
        if self.is_log_scale: ax.set_yscale('log')
        ax.axis('off') # Turn off axis. 
        fig.canvas.draw()
        self._spectrum = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).\
                                       reshape(fig.canvas.get_width_height()[::-1] + (3,)) # Save the plot from buffer as a numpy array. 
                                       
        if visualize: plt.show()
        if save and self.spectrum_fig_path is not None: 
            plt.savefig(self.spectrum_fig_path, bbox_inches='tight', pad_inches=0)
        
        plt.close()
        plt.cla()
        plt.clf()
        

class Audio(object):
    """
    """

    def __init__(self, file_path=None, data=None, data_label=None, sr=ACOUSTIC.AUDIO_SAMPLING_RATE, omit=0.):
        """
        """

        self.file_path = file_path
        self.sampling_rate = sr
        self.omit_duration = omit # Unit: In s.

        self.audio_file_name = data_label if data_label is not None else self._get_audio_name()
        self.data = data if data is not None and self.file_path is None else \
                    self._read_data_from_path()
        self.audio_len = len(self.data)
    

    def __len__(self):
        """
        """

        return self.audio_len


    def _get_audio_name(self):
        """
        """

        if self.file_path is not None: self.audio_file_name = Path(self.file_path).stem
        else: self.audio_file_name = "audio_0" # Default name of audio files' folder. 


    def _read_data_from_path(self):
        """
        """

        if self.file_path is None: raise ValueError("Audio data not found. ")
        else: 
            file_extension = os.path.splitext(self.file_path)[1]
            if file_extension == ".wav": 
                y, _ = librosa.load(self.file_path, sr=self.sampling_rate, offset=self.omit_duration)
            elif file_extension == ".npy": 
                y = np.load(self.file_path)
            else: raise ValueError("Audio file format not recognizable. ")

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


if __name__ == "__main__":
    """
    """

    pass