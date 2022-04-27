# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:22:09 2022

@author: hlinl
"""


import os
import copy
import sys
DIR_ABS = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR_ABS))

import numpy as np
import matplotlib.image as mig

from PARAM.IMG import *


def shifting(data_matrix, axis, origin, target):
    """
    Shift the origin of new basis coordinate system to mean point of the data. 
    Average along with axis-1. 

    Parameters:
    ----------
        data_matrix: 2D Array. 
            Size: nFeatures x nSamples.

    Returns:
    ----------
        data_new: 2D Array with the same size as data_matrix. 
            Mean-shifted data. 
        mean_vect: 1D Array of float. 
            The mean value of each feature. 
    """
    
    if axis == 0: data_new = copy.deepcopy(data_matrix)
    elif axis == 1: data_new = copy.deepcopy(data_matrix.T)
    else: data_new = copy.deepcopy(origin - target)

    data_new += (target - origin)

    if axis == 1: return data_new.T
    else: return data_new


def normalization(data_matrix, axis, mode='unitize'):
    """
    """
    
    if mode == 'unitize':
        if axis == 1: data_matrix_normalized = copy.deepcopy(data_matrix.T)
        else: data_matrix_normalized = copy.deepcopy(data_matrix)

        norm_vect = np.linalg.norm(data_matrix_normalized, axis=0)
        data_matrix_normalized /= norm_vect

        if axis == 1: return data_matrix_normalized.T
        else: return data_matrix_normalized

    else: return data_matrix


def standardization(matrix, axis, mean_vect, std_vect):
    """
    """
    
    if axis == 1: matrix_standardized = copy.deepcopy(matrix.T)
    else: matrix_standardized = copy.deepcopy(matrix)

    matrix_standardized = (matrix_standardized - mean_vect) / std_vect

    if axis == 1: return matrix_standardized.T
    else: return matrix_standardized


def eigenDecomposition(square_matrix):
    """
    """
    
    return np.linalg.eig(square_matrix)


def angle_2vect(vect_1, vect_2):
    """
    Both shape = (-1,).

    Return the value in degrees. 
    """

    cos_theta = np.dot(vect_1, vect_2) / (np.linalg.norm(vect_1)*np.linalg.norm(vect_2))
    theta_rad = np.arccos(cos_theta)

    return theta_rad * 180. / np.pi # [0., 180.]

