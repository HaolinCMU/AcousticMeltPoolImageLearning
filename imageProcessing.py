# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 00:14:04 2021

@author: hlinl
"""

import os
import copy

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mig
import librosa

from PARAM_ML import *
from PARAM_IMG import *
from dataprocessing.moments import *


class Image(object):
    """
    Generate an image object from a given image file path. 
    Inherited from 'object'. 
    """

    def __init__(self, file_path=None, image_matrix=np.array([None])):
        """
        Class initialization function. 

        Parameters:
        ----------
            file_path: String. 
                The path of a image file. 
                Default: None. 
            image_matrix: 2D Float array. Dimension: N x M.  
                The two-dimensional matrix of the image. Value range: (0-1).
                Take higher priority than 'file_path' as input. 
                Default: np.array([None]).  
        
        Attribute:
        ----------
            self.file_path: String. 
                Same as the parameter 'file_path'. 
            self.image_matrix: 2D Float matrix. 
                The matrix version of the image, either from the file specified by 'self.file_path' or from the given 'image_matrix'. 
            self.image_width: Int. 
                The pixel dimension of the image matrix's axis-1 (horizontal). 
            self.image_length: Int. 
                The pixel dimension of the image matrix's axis-0 (vertical).
        """

        self.file_path = file_path

        self.image_matrix = None
        if image_matrix.any() == None and self.file_path != None:
            self.image_matrix = mig.imread(self.file_path)
        if image_matrix.any() != None:
            self.image_matrix = image_matrix

        self.image_width = self.image_matrix.shape[1]
        self.image_length = self.image_matrix.shape[0]

    
    def get_pixelNum(self):
        """
        """

        return int(self.image_length*self.image_width)


    def get_length(self):
        """
        """
        
        return self.image_length

    
    def get_width(self):
        """
        """
        
        return self.image_width


class MeltPoolImageProcessor(Image):
    """
    """

    def __init__(self, file_path=None, image_matrix=np.array([None]), 
                 meltpool_intensity_threshold=(0.8,1.0)):
        """
        """

        super(MeltPoolImageProcessor, self).__init__(file_path, image_matrix)
        self.intensity_threshold = meltpool_intensity_threshold
        self.meltpool_pixel_index_array = None
        self.meltpool_filtered_image = None

        self._extract_meltpool_pixel_indices()
        self._filter_meltpool_image()
    
    
    def _extract_meltpool_pixel_indices(self):
        """
        Extract the indices of pixels with intensity values fallen in between the range defined by `self.intensity_threshold`. 
        """

        indices_array_row_col = np.where(np.logical_and(self.image_matrix >= self.intensity_threshold[0], 
                                                        self.image_matrix <= self.intensity_threshold[1]))

        self.meltpool_pixel_index_array = np.vstack((indices_array_row_col[0], indices_array_row_col[1])).T # [row | col]. 

    
    def _filter_meltpool_image(self):
        """
        """
        
        meltpool_filtered_image_matrix = np.zeros(shape=self.image_matrix.shape)
        
        for i in range(self.meltpool_pixel_index_array.shape[0]):
            row_ind_temp = self.meltpool_pixel_index_array[i,0]
            col_ind_temp = self.meltpool_pixel_index_array[i,1]
            
            meltpool_filtered_image_matrix[row_ind_temp,col_ind_temp] = self.image_matrix[row_ind_temp,col_ind_temp]
        
        self.meltpool_filtered_image = copy.deepcopy(meltpool_filtered_image_matrix)

    
    def _get_meltpool_pixel_indices(self):
        """
        Get the indices of pixels with intensity values fallen in between the range defined by `self.intensity_threshold`. 
        """ 

        return self.meltpool_pixel_index_array
    
    
    def _get_meltpool_filtered_image(self):
        """
        """
        
        return self.meltpool_filtered_image
        

    def _Hu_moment_invariant_computation(self, feature_num=2):
        """
        """
        
        

        pass


    def get_meltpool_area(self):
        """
        """

        return self.meltpool_pixel_index_array.shape[0]


    def get_Hu_Moments(self):
        pass


    def get_Zernike_Moments(self):
        pass
    

