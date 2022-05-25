# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:45:17 2022

@author: hlinl
"""


import os
import copy
import sys
DIR_ABS = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR_ABS))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .imgBasics import *
from .utility import *

from PARAM import *


DEBUG = False # Default: False. 


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

    def __init__(self, file_path=None, image_matrix_filtered=np.array([None]), m_max=4):
        """
        Input image must be centered. 
        """

        super(ZernikeMoments, self).__init__(file_path, image_matrix_filtered)
        self.m_max = m_max
        self._m_n_dict = {}
        self._center_pt = np.array([self.image_length/2., self.image_width/2.])
        self._reconstruct_intensity_threshold = 60000

        self._set_m_n_dict()
        self.r_matrix, self._r_max = self._r_matrix(self.image_matrix, self._center_pt)
        self.theta_matrix = self._theta_matrix(self.image_matrix, self._center_pt)
        
        self.Zernike_moments_dict = {}
        self.zernike_moments_array = None
        self._V_dict = {} # Dictionary of Zernike polynomial. Similar to Eigenfaces. 
        
        self._compute_Zernike_moments()


    @staticmethod
    def FACT(n):
        """
        n should be an integer. 
        """

        return np.math.factorial(n)

    
    @staticmethod
    def _conjugate(z):
        """
        """

        return np.conjugate(z)

    
    @staticmethod
    def _indices(matrix):
        """
        matrix should be 2D. 
        """

        indices = np.where(matrix>=0.)

        return np.stack((indices[0].reshape(matrix.shape), 
                         indices[1].reshape(matrix.shape)), axis=2).astype(int)
    
    
    @staticmethod
    def _binarize_with_bar(image_matrix, threshold=0.8):
        """
        """
        
        return (image_matrix>=threshold).astype(float)

    
    def _set_m_n_dict(self):
        """
        """

        for i in range(self.m_max+1):
            search_range = np.arange(i+1)
            
            if i % 2 == 0: 
                n_list_temp = list(set(list(search_range[::2])+list(-search_range[::2])))
            else: 
                n_list_temp = list(set(list(search_range[1::2])+list(-search_range[1::2])))
            
            self._m_n_dict[i] = copy.deepcopy([(i,n) for n in n_list_temp])


    def _r_matrix(self, image_matrix, origin):
        """
        origin: (2,)
        """

        l_0, l_1 = image_matrix.shape

        row_indices_array = self._indices(image_matrix)[:,:,0]
        col_indices_array = self._indices(image_matrix)[:,:,1]
        
        row_dist_array = row_indices_array - origin[0]
        col_dist_array = col_indices_array - origin[1]
        
        r_matrix = copy.deepcopy(np.sqrt(row_dist_array**2 + col_dist_array**2))
        r_matrix /= np.max(r_matrix)

        return r_matrix, np.max(r_matrix)
    

    def _theta_matrix(self, image_matrix, origin):
        """
        origin: (2,)
        Theta axes: same as image matrix axes. 
        """

        l_0, l_1 = image_matrix.shape

        row_indices_array = self._indices(image_matrix)[:,:,0]
        col_indices_array = self._indices(image_matrix)[:,:,1]

        row_dist_array = row_indices_array - origin[0]
        col_dist_array = col_indices_array - origin[1]

        return np.arctan2(col_dist_array, row_dist_array)


    def _R_matrix(self, m, n):
        """
        m, n should be integers. 
        """

        m, n = int(m), int(n)

        iter_num, R_mn_matrix = int((m-abs(n))/2 + 1), np.zeros(shape=self.r_matrix.shape)
        for i in range(iter_num):
            R_matrix_temp = (-1)**i*self.FACT(m-i)/(self.FACT(i)*self.FACT(int((m+abs(n))/2-i))* \
                                                    self.FACT(int((m-abs(n))/2-i)))*self.r_matrix**(m-2*i)
            R_mn_matrix += R_matrix_temp

        return R_mn_matrix

    
    def _Phi_matrix(self, n):
        """
        """

        return np.exp(n*np.vectorize(complex)(np.zeros(shape=self.theta_matrix.shape), self.theta_matrix))


    def _V_matrix(self, m, n):
        """
        m, n: integers. 
        """

        R_mn_matrix = self._R_matrix(m, n)
        Phi_n_matrix = self._Phi_matrix(n)

        V_mn_matrix = R_mn_matrix * Phi_n_matrix
        
        self._V_dict[(m,n)] = copy.deepcopy(V_mn_matrix)

        return copy.deepcopy(V_mn_matrix)


    def _Z(self, m, n):
        """
        center: the image center. 
        """

        Z_mn = copy.deepcopy(np.sum(self.image_matrix*self._conjugate(self._V_matrix(m, n))))
        Z_mn *= (m+1) / np.pi

        return Z_mn


    def _compute_Zernike_moments(self):
        """
        """

        for _, val in self._m_n_dict.items():
            for m_and_n in val:
                m, n = m_and_n
                Z_mn = self._Z(m, n)

                self.Zernike_moments_dict[(m,n)] = Z_mn

        self.zernike_moments_array = copy.deepcopy(list(self.Zernike_moments_dict.values()))
        
    
    def Zernike_moments(self):
        """
        """
        
        return self.zernike_moments_array

    
    def reconstruct(self, zernike_moments_dict):
        """
        Following the order of Noll's sequential indices. 
        """

        image_reconstructed = np.zeros(shape=self.r_matrix.shape).astype(complex)
        for key, Z_mn in zernike_moments_dict.items():
            V_mn_matrix = self._V_dict[key]
            image_reconstructed += copy.deepcopy(Z_mn*V_mn_matrix)
            
        return copy.deepcopy(image_reconstructed.real)
        
        # return copy.deepcopy(self._binarize_with_bar(image_reconstructed.real, 
        #                                               threshold=self._reconstruct_intensity_threshold))

    
    def _rotational_invariance_validation(self):
        """
        """

        

        pass


if __name__ == "__main__":
    # file_path = 'C:/Users/hlinl/OneDrive/Desktop/New folder/Data/raw_image_data/Layer042_Section_08_S0001/Layer042_Section_08_S0001000926.png'
    # file_path = 'C:/Users/hlinl/OneDrive/Desktop/zernike/F/gl5un.png'
    # m_max_list = [15]
    
    # for ind, m_max in enumerate(m_max_list):
    #     zernike = ZernikeMoments(file_path=file_path, m_max=m_max)
    #     zernike_moments = zernike.Zernike_moments()
    #     img_reconstruct = zernike.reconstruct(zernike.Zernike_moments_dict).reshape(zernike.image_matrix.shape)

    #     if ind == 0:
    #         plt.figure(figsize=(20,20))
    #         plt.rcParams.update({"font.size": 35})
    #         plt.tick_params(labelsize=35)
    #         plt.imshow(zernike.image_matrix, cmap='gray')
    #         plt.savefig("ori.png")
            
    #     plt.figure(figsize=(20,20))
    #     plt.rcParams.update({"font.size": 35})
    #     plt.tick_params(labelsize=35)
    #     plt.imshow(copy.deepcopy(img_reconstruct), cmap='gray')
    #     plt.savefig("m_{}.png".format(m_max))

    pass
