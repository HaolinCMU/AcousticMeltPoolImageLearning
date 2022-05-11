# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 00:19:37 2022

@author: hlinl
"""


import os
import copy
import cv2
import sys
DIR_ABS = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR_ABS))

import numpy as np
import matplotlib.image as mig

from PARAM.IMG import *


def rotate_2D(data_matrix, theta, center_pt):
    """
    data_matrix: shape=(n_points, n_features=2)
    theta: in degrees. 
    center_pt: shape=(n_features=2,)
    """

    theta_rad = theta * np.pi / 180.
    rot_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                           [np.sin(theta_rad), np.cos(theta_rad)]])

    vector_matrix = data_matrix - center_pt
    vector_matrix_rotated = copy.deepcopy(rot_matrix @ vector_matrix.T).T

    return vector_matrix_rotated + center_pt


def projection_along_axis(pixel_index_array, origin_pt, axis_vect):
    """
    `pixel_index_array`: shape=(n_sample, n_feature). 
    `origin_pt`: shape=(n_feature,). 
    `axis_vect`: shape=(n_feature,). 
    """
    
    vect_array = copy.deepcopy(pixel_index_array - origin_pt)
    inner_product_array = copy.deepcopy(vect_array @ axis_vect.reshape(-1,1)).reshape(-1)
    
    return inner_product_array


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

        self.image_length, self.image_width = self.image_matrix.shape

        self._default_background_value = 0. # 0 or a very small value, e.g. 1e-5. 

    
    def pixelNum(self):
        """
        """

        return int(self.image_length*self.image_width)


    def length(self):
        """
        """
        
        return self.image_length

    
    def width(self):
        """
        """
        
        return self.image_width

    
    def blank_version(self):
        """
        """

        return np.ones(shape=(self.image_length, self.image_width)) * self._default_background_value
    

    def refine_asper_threshold(self, image_matrix, threshold):
        """
        """
        
        image_matrix_refined = copy.deepcopy(image_matrix)
        image_matrix_refined[image_matrix_refined < threshold] = self._default_background_value
        
        return image_matrix_refined