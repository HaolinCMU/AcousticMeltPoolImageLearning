# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:45:17 2022

@author: hlinl
"""


import os
import cv2
import numpy as np

from .imgBasics import *
from .utility import *


class HuMoments(Image):
    """
    """

    def __init__(self, image_matrix=np.array([None]), img_file_path=None):
        """
        """

        super(HuMoments, self).__init__(img_file_path, image_matrix)
        self.Hu_moments_array = None

        self._compute_Hu_moments()


    def _compute_Hu_moments(self):
        """
        """
        
        moments = cv2.moments(self.image_matrix)
        Hu_moments_array = cv2.HuMoments(moments)
        # Hu_moments_array = np.log(np.abs(Hu_moments_array))
        
        self.Hu_moments_array = Hu_moments_array.reshape(-1)


    def Hu_moments(self):
        """
        """

        return self.Hu_moments_array

    
    def _invariance_validation(self):
        """
        """

        pass


class ZernikeMoments(Image):
    """
    """

    def __init__(self, file_path=None, image_matrix_filtered=np.array([None])):
        """
        """

        super(ZernikeMoments, self).__init__(file_path, image_matrix_filtered)
        self.Zernike_moments_array = None

        self._compute_Zernike_moments()


    def _compute_Zernike_moments(self):
        """
        """
        
        moments = cv2.moments(self.image_matrix)
        Hu_moments_array = cv2.HuMoments(moments)
        # Hu_moments_array = np.log(np.abs(Hu_moments_array))
        
        self.Zernike_moments_array = Hu_moments_array.reshape(-1)


    def Zernike_moments(self):
        """
        """

        return self.Zernike_moments_array

    
    def _invariance_validation(self):
        """
        """

        pass