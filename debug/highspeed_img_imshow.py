# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:33:25 2022

@author: hlinl
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mig
import PIL


img_dir = 'F:/data/raw/highspeed/Layer0221_P330_V0250_C001H001S0001/Images_S0001009091.png'

plt.figure()
image_matrix = mig.imread(img_dir)
# plt.imshow(image_matrix, cmap='gray')