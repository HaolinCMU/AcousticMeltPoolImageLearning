# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:03:30 2022

@author: hlinl
"""

import numpy as np
from image_processing import *


visual_mean, visual_std = visual_data_standard(visual_dir="C:/Users/hlinl/Desktop/visual_median_standard")
np.save("C:/Users/hlinl/Desktop/visual_mean_vect.npy", visual_mean)
np.save("C:/Users/hlinl/Desktop/visual_std_vect.npy", visual_std)