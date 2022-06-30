# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 22:47:38 2022

@author: hlinl
"""


import os
import copy
import sys
DIR_ABS = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR_ABS))

import numpy as np

from .imgBasics import *
from .utility import *
from PARAM.IMG import *


class PCA(object):
    """
    """

    def __init__(self, training_matrix, PC_num, mode='normal'):
        """
        training_matrix with Axis-0 be the sample axis. 
        mode: 'normal' (default) or 'transpose'. 
            'normal' (default): decompose on sample axis. 
            'transpose': decompose on feature axis. 
        """

        self.mode = mode
        self.PC_num = PC_num

        if self.mode == 'normal': self.matrix = training_matrix
        elif self.mode == 'transpose': self.matrix = training_matrix.T
        else: self.matrix = np.eye(training_matrix.shape[0])
        
        self._mean_vect = None
        self._eigFace_matrix = None
        self._weights = None
        self._eigFace_matrix_full = None
        self._weights_full = None

        self._set_encoder_decoder()


    def _meanshifting(self):
        """
        """

        if self.mode == 'transpose': axis = 1
        else: axis = 0

        self._mean_vect = np.mean(self.matrix, axis=axis).reshape(-1)

        return shifting(self.matrix, axis=axis, origin=self._mean_vect, 
                        target=np.zeros(shape=self._mean_vect.shape))


    @staticmethod
    def _eigendecomposing(matrix):
        """
        """

        cov_matrix = copy.deepcopy(matrix @ matrix.T)
        return eigenDecomposition(cov_matrix)

    
    def _eigSorting(self, eigVal, eigVect):
        """
        """

        eigFace_num = eigVal.shape[0]
        eigVal_sorted = np.zeros(shape=eigVal.shape, dtype=complex)
        eigVect_sorted = np.zeros(shape=eigVect.shape, dtype=complex)

        eigVal_sorted_indices = np.argsort(np.real(eigVal))
        eigVal_PC_indices = eigVal_sorted_indices[-1:-(eigFace_num+1):-1] # Pick PC_num indices of largest principal eigenvalues
        
        for i, index in enumerate(eigVal_PC_indices): # From biggest to smallest
            eigVal_sorted[i] = eigVal[index] # Pick PC_num principal eigenvalues. Sorted. 
            eigVect_sorted[:,i] = eigVect[:,index] # Pick PC_num principal eigenvectors. Sorted. 

        return np.real(eigVect_sorted)


    def _update_eigFaces_weights(self):
        """
        """
        
        self._eigFace_matrix = copy.deepcopy(self._eigFace_matrix_full[:,:self.PC_num])
        self._weights = copy.deepcopy(self._weights_full[:,:self.PC_num])


    def _set_encoder_decoder(self):
        """
        """

        matrix_meanshifted = self._meanshifting()
        eigVal, eigVect = self._eigendecomposing(matrix_meanshifted)

        eigVect_sorted = self._eigSorting(eigVal, eigVect) # Real matrix. 

        if self.mode == 'normal':
            eigFace_matrix = copy.deepcopy(self.matrix.T @ eigVect_sorted)
            self._eigFace_matrix_full = normalization(eigFace_matrix, axis=0)
            self._weights_full = self.matrix @ self._eigFace_matrix_full # sample_num * PC_num. (max_PC_num=sample_num)
        elif self.mode == 'transpose':
            self._eigFace_matrix_full = copy.deepcopy(eigVect_sorted)
            self._weights_full = self.matrix.T @ self._eigFace_matrix_full # sample_num * PC_num. . (max_PC_num=feature_num)
        else: 
            self._eigFace_matrix_full = np.zeros(shape=eigVect.shape)
            self._weights_full = np.zeros(shape=eigVect.shape)

        self._update_eigFaces_weights()


    def eigFaces(self):
        """
        """

        return self._eigFace_matrix


    def weights(self):
        """
        """

        return self._weights


    def change_PC_num_to(self, new_PC_num):
        """
        Merely change the `PC_num`. 
        """

        self.PC_num = new_PC_num
        self._update_eigFaces_weights()

    
    def encoding(self, input_matrix):
        """
        input_matrix with Axis-0 be the sample axis. 
        """

        input_matrix_meanshifted = shifting(input_matrix, axis=0, origin=self._mean_vect, 
                                            target=np.zeros(shape=self._mean_vect.shape))
        return copy.deepcopy(input_matrix_meanshifted @ self._eigFace_matrix) # sample_num_test # weight_num
    

    def reconstructing(self, weight_matrix):
        """
        """

        data_matrix_reconstructed = copy.deepcopy(weight_matrix @ self._eigFace_matrix.T)
        return copy.deepcopy(shifting(data_matrix_reconstructed, axis=0, 
                                      origin=np.zeros(shape=self._mean_vect.shape), 
                                      target=self._mean_vect))
