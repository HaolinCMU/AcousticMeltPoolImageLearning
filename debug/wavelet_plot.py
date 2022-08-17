# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:12:47 2022

@author: hlinl
"""


import numpy as np
import matplotlib.pyplot as plt
import librosa
import pywt


def plot_wavelet(time, signal, scales, waveletname = 'cmor', 
                 cmap = plt.cm.seismic, title = 'Wavelet Transform (Power Spectrum) of signal', 
                 ylabel = 'Period (years)', 
                 xlabel = 'Time'):
    """
    """
    
    
    dt = time[1] - time[0]
    # [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    # levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    # contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(figsize=(10,10)) 
    # im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    im = ax.contourf(np.log2(power), extend='both',cmap=cmap)
    ax.invert_yaxis()
    ax.axis('off')
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    """
    """
    
    sr = 96e3
    sample_length = 128
    offset = 0.0720
    
    wavelet = 'cmor'
    scales = np.arange(1,31)
    
    start_ind = 64
    path = 'C:/Users/hlinl/Desktop/acoustic_image_learning/data/raw_audio_data/Layer003_Section_01_S0001.wav'

    
    y, _ = librosa.load(path, sr=sr, offset=offset)
    
    signal = y[start_ind:start_ind+sample_length]
    time = np.arange(0, len(signal)) * (1./sr)
    
    plot_wavelet(time, signal, scales=scales)
    
    
    