# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 19:47:36 2022

@author: hlinl
"""


import copy
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mig
import numpy as np

image_path_list = glob.glob("*.png")
intensity_threshold = (0.8, 1.0) # Default: (0.8, 1.0). 

for image_name in image_path_list:
    save_image_name = '_'.join(image_name.split('_')[:2])
    img_matrix_temp = mig.imread(image_name)
    
    indices_array_row_col = np.where(np.logical_and(img_matrix_temp >= intensity_threshold[0], 
                                                    img_matrix_temp <= intensity_threshold[1])) # Float tuple. (row indices, col_indices)
    
    bright_pixel_index_array_temp = np.vstack((indices_array_row_col[1], indices_array_row_col[0])).T # [ col_ind | row_ind ]
    # bright_pixel_index_list_temp = np.vstack((indices_array_row_col[0], indices_array_row_col[1])).T.tolist()
    
    plt.figure()
    plt.imshow(img_matrix_temp, cmap='gray')
    plt.scatter(bright_pixel_index_array_temp[:,0], 
                bright_pixel_index_array_temp[:,1], 
                c='r', s=1.0)
    plt.savefig("{}_bright_show.png".format(save_image_name))
    

    
    




