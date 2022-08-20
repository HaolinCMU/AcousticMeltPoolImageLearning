# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:32:54 2022

@author: hlinl
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import os
from tkinter import filedialog
from tkinter import *

def set_path():      
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder_selected = filedialog.askopenfilename(parent=root)
    return folder_selected

filename = set_path()

try:
    data = np.load(filename)
except:
    data = np.loadtxt(filename)

print(data.shape)    
    
    
plt.figure()
plt.plot(np.arange(0,data.shape[1])/100000,data[0,:],label='chan1')
plt.plot(np.arange(0,data.shape[1])/100000,data[1,:],label='chan2',alpha=0.5)
plt.legend()
plt.ylim([-10,10])
