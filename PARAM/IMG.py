# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 01:19:34 2022

@author: hlinl
"""

import os
import numpy as np

INTENSITY_THRESHOLD = (0.8, 1.0)

# DBSCAN PARAM
DBSCAN_EPSILON = 2
DBSCAN_MIN_PTS = 5

# PCA PARAM - one frame. 
PC_NUM_FRAME = 2
PCA_MODE_FRAME = 'transpose'
FRAME_ALIGN_MODE = 'principal' # 'principal' or 'secondary' or 'other'. 
FRAME_REALIGN_AXIS_VECT = np.array([0.,1.])