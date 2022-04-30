# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 00:14:04 2021

@author: hlinl
"""


import os
import glob
import copy

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mig

from cmath import nan
from scipy.stats import skew, kurtosis
from scipy.ndimage import rotate

from PARAM.IMG import *
from PARAM.BASIC import *
from dataprocessing import *


class Frame(imgBasics.Image):
    """
    """

    def __init__(self, file_path=None, image_matrix=np.array([None]), 
                 intensity_threshold=INTENSITY_THRESHOLD):
        """
        """

        super(Frame, self).__init__(file_path, image_matrix)
        self.intensity_threshold = intensity_threshold
        self._isEmpty = False # Boolean. True if no pixels satisfy the intensity threshold. 

        self._bright_pixel_index_array = np.array([[None, None]])
        self._bright_pixel_intensity_array = np.array([None]) # Same order as `self._bright_pixel_index_array`. 
        self.visible_image = np.array([None])
        self.mass_center_pt = np.array([None]) # 1D Int array. A pixel. 
        self._total_area = None

        self.meltpool_pixel_index_array = np.array([[None, None]])
        self.meltpool_pixel_intensity_array = np.array([None]) # Same order as `self.meltpool_pixel_intensity_array`.
        self._meltpool_image = np.array([None])
        self.meltpool_center_pt = np.array([None]) # 1D Int array. A pixel. 
        self._meltpool_area = None # Int. Number of melt pool pixels. 
        self._meltpool_length = None # Float. 
        self._meltpool_width = None # Float. 
        self._meltpool_principal_axes = np.array([None]) # Float. 

        self._meltpool_spatial_skewness = None # Float. 
        self._meltpool_intensity_kurtosis = None # Float.
        self._total_intensity_kurtosis = None # Float 

        self.visible_image_straightened = np.array([None])
        self._visible_pixel_index_straightened = np.array([[None, None]])
        self.meltpool_image_straightened = np.array([None])
        self._meltpool_pixel_index_straightened = np.array([[None, None]])

        self._thresholding()
        self._set_meltpool()
    

    def total_area(self):
        """
        """

        return self._total_area


    def meltpool_area(self):
        """
        """

        return self._meltpool_area


    def visible_pixel_indices(self):
        """
        Get the indices of pixels with intensity values fallen in between the range defined by `self.intensity_threshold`. 
        """ 

        return self._bright_pixel_index_array


    def meltpool_pixel_indices(self):
        """
        Get the indices of pixels with intensity values fallen in between the range defined by `self.intensity_threshold`. 
        """ 

        return self.meltpool_pixel_index_array
    
    
    def thresheld_image(self):
        """
        """
        
        return self.visible_image

    
    def meltpool_image(self):
        """
        """
        
        return self._meltpool_image
    
    
    def meltpool_length(self):
        """
        """
        
        return self._meltpool_length
    
    
    def meltpool_width(self):
        """
        """
        
        return self._meltpool_width
    

    @staticmethod
    def _dimension_along_axis(pixel_index_array, origin_pt, axis_vect):
        """
        `pixel_index_array`: shape=(n_sample, n_feature). 
        `origin_pt`: shape=(n_feature,). 
        `axis_vect`: shape=(n_feature,). 
        """
        
        inner_product_array = imgBasics.projection_along_axis(pixel_index_array, origin_pt, axis_vect)
        
        return max(inner_product_array) - min(inner_product_array)
    
    
    def _set_meltpool_dimensions(self):
        """
        """
        
        if not self._isEmpty:
            self._meltpool_length = self._dimension_along_axis(self.meltpool_pixel_index_array, self.meltpool_center_pt, 
                                                               copy.deepcopy(self._meltpool_principal_axes[:,0]))
            self._meltpool_width = self._dimension_along_axis(self.meltpool_pixel_index_array, self.meltpool_center_pt, 
                                                              copy.deepcopy(self._meltpool_principal_axes[:,1]))
        
        else:
            self._meltpool_length = 0.
            self._meltpool_width = 0.


    @staticmethod
    def _specify_image(image, pixel_index_array, img_val_array):
        """
        `pixel_index_array`: shape=(n_sample, n_feature)
        `img_val_array`: shape=(n_sample,), following the same order. 
        """

        image_specified = copy.deepcopy(image)

        for i in range(pixel_index_array.shape[0]):
            row_ind_temp, col_ind_temp = pixel_index_array[i,:].astype(int)
            image_specified[row_ind_temp,col_ind_temp] = img_val_array[i]
        
        return image_specified


    def _filter_image(self, pixel_index_array):
        """
        """

        image_matrix = copy.deepcopy(self.image_matrix)
        filtered_image_matrix = self.blank_version()
        
        for i in range(pixel_index_array.shape[0]):
            row_ind_temp, col_ind_temp = pixel_index_array[i,:].astype(int)

            filtered_image_matrix[row_ind_temp,col_ind_temp] = image_matrix[row_ind_temp,col_ind_temp]
        
        return copy.deepcopy(filtered_image_matrix)
    
    
    def _mask_image(self, pixel_index_array):
        """
        """

        filtered_image_matrix = self.blank_version()
        
        for i in range(pixel_index_array.shape[0]):
            row_ind_temp, col_ind_temp = pixel_index_array[i,:].astype(int)
            
            filtered_image_matrix[row_ind_temp,col_ind_temp] = 1
        
        return copy.deepcopy(filtered_image_matrix)


    def binarize(self, image_matrix):
        """
        Used for contour extraction. 
        """
        
        indices_array_row_col = np.where(image_matrix>self._default_background_value)
        
        bright_pixel_index_array = np.vstack((indices_array_row_col[0], 
                                              indices_array_row_col[1])).astype(int).T # [row | col]; [Axis-0 | Axis-1].
        
        return self._mask_image(bright_pixel_index_array)


    def _centering(self, pixel_index_array, center_pt):
        """
        Move `pixel_index_array` from its center to the center of the frame. 

        center_pt: the center of `pixel_index_array` before centering. 
        """

        center_pt = copy.deepcopy(center_pt.astype(int).reshape(-1))
        center_pt_blank_version = np.array([self.length()/2., self.width()/2.]).astype(int).reshape(-1)
        pixel_index_array_translated = utility.shifting(pixel_index_array, axis=0, 
                                                        origin=center_pt, target=center_pt_blank_version).astype(int)

        return pixel_index_array_translated


    @staticmethod
    def _rotate(image_matrix, theta):
        """
        pixel_index_array: shape=(n_points, n_features)
        theta: in degrees. 
        center_pt: shape=(n_features,)
        """
        
        return rotate(image_matrix, theta, reshape=False)


    def _thresholding(self):
        """
        Extract the indices of pixels with intensity values fallen in between the range defined by `self.intensity_threshold`. 
        """

        indices_array_row_col = np.where(np.logical_and(self.image_matrix >= self.intensity_threshold[0], 
                                                        self.image_matrix <= self.intensity_threshold[1]))
        self._bright_pixel_index_array = np.vstack((indices_array_row_col[0], 
                                                    indices_array_row_col[1])).astype(int).T # [row | col]; [Axis-0 | Axis-1].

        if self._bright_pixel_index_array.shape[0] != 0:
            self._isEmpty = False
            self._bright_pixel_intensity_array = copy.deepcopy(np.array([self.image_matrix[i,j] 
                                                                         for i,j in self._bright_pixel_index_array]).reshape(-1))
            self.visible_image = self._filter_image(self._bright_pixel_index_array)
            self.mass_center_pt = self._get_center_of(self._bright_pixel_index_array)

        else:
            self._isEmpty = True
            self._bright_pixel_index_array = np.array([[None, None]])
            self._bright_pixel_intensity_array = np.array([None])
            self.visible_image = self.blank_version()
            self.mass_center_pt = np.array([self.length()/2., self.width()/2.]).astype(int).reshape(-1)


    def _set_meltpool(self):
        """
        Extract the pixel indices of melt pool, which is the largest pixel cluster in the filtered image. 
        """

        if not self._isEmpty:
            pixel_clusters = clustering.dbscan(self._bright_pixel_index_array, DBSCAN_EPSILON, DBSCAN_MIN_PTS)
            self.meltpool_pixel_index_array = pixel_clusters.largest_cluster().astype(int)
            self.meltpool_pixel_intensity_array = copy.deepcopy(np.array([self.image_matrix[i,j] 
                                                                          for i,j in self.meltpool_pixel_index_array]).reshape(-1))
            self._meltpool_image = self._filter_image(self.meltpool_pixel_index_array)
            self.meltpool_center_pt = pixel_clusters.center_pt(self.meltpool_pixel_index_array).astype(int)

        else:
            self.meltpool_pixel_index_array = np.array([[None, None]])
            self.meltpool_pixel_intensity_array = np.array([None])
            self._meltpool_image = self.blank_version()
            self.meltpool_center_pt = copy.deepcopy(self.mass_center_pt)
        
        self._total_area = self._bright_pixel_index_array.shape[0]
        self._meltpool_area = self.meltpool_pixel_index_array.shape[0]
        
        self._set_meltpool_principal_axes()
        self._set_meltpool_dimensions()

    
    def _set_meltpool_principal_axes(self):
        """
        """

        if not self._isEmpty: 
            pca_coder = pca.PCA(self.meltpool_pixel_index_array.astype(float), 
                                PC_num=PC_NUM_FRAME, mode=PCA_MODE_FRAME)
            self._meltpool_principal_axes = pca_coder.eigFaces()
        else:
            self._meltpool_principal_axes = np.array([[None, None], [None, None]])

        
    def _meltpool_axis_of(self, keyword='principal'):
        """
        """
        
        meltpool_axis = np.array([None, None])
        
        if not self._isEmpty:
            if keyword == 'principal': 
                meltpool_axis = copy.deepcopy(self._meltpool_principal_axes[:,0]).reshape(-1)
            elif keyword == 'secondary':
                meltpool_axis = copy.deepcopy(self._meltpool_principal_axes[:,1]).reshape(-1)
            else: pass
            
        else: pass
        
        return meltpool_axis

    
    def _compute_meltpool_spatial_skewness(self, axis):
        """
        axis: shape=(-1,)
        """

        if self._isEmpty: self._meltpool_spatial_skewness = None

        else:             
            inner_product_array = imgBasics.projection_along_axis(self.meltpool_pixel_index_array, 
                                                                  self.meltpool_center_pt, axis)
            self._meltpool_spatial_skewness = skew(inner_product_array)
        
        return self._meltpool_spatial_skewness
    

    def _compute_intensity_kurtosis(self, keyword='meltpool'): 
        """
        """

        kurtosis_val = None

        if keyword == 'meltpool':
            if not self._isEmpty: 
                kurtosis_val = kurtosis(self.meltpool_pixel_intensity_array)
            else: pass

            self._meltpool_intensity_kurtosis = kurtosis_val

        elif keyword == 'total':
            if not self._isEmpty: 
                kurtosis_val = kurtosis(self._bright_pixel_intensity_array)
            else: pass

            self._total_intensity_kurtosis = kurtosis_val

        else: pass
    
        return kurtosis_val
    
   
    def _set_straighten_angle(self, meltpool_axis, align_axis=np.array([0.,1.])):
        """
        """
        
        rotation_angle = 0.
        
        if not self._isEmpty:
            skewness = self._compute_meltpool_spatial_skewness(meltpool_axis)
            
            if skewness <= 0.: axis_vect = copy.deepcopy(meltpool_axis)
            else: axis_vect = -copy.deepcopy(meltpool_axis)

            rotation_angle = utility.angle_2vect(axis_vect, align_axis)
            
            if np.sign(np.cross(align_axis, axis_vect)) >= 0.: rotation_angle *= -1
            else: pass

        else: pass
        
        return rotation_angle
    

    def _center_rotate_image(self, pixel_index_array, pixel_val_array, 
                             self_axis, align_axis=np.array([0.,1.])):
        """
        Always meltpool center. 
        """
        
        pixel_index_array_centered = self._centering(pixel_index_array, self.meltpool_center_pt)
        centered_version = self._specify_image(self.blank_version(), pixel_index_array_centered, pixel_val_array)
        
        theta = self._set_straighten_angle(meltpool_axis=self_axis, align_axis=align_axis)
        straightened_version = copy.deepcopy(self._rotate(centered_version, theta))
        
        return straightened_version


    def straighten(self, ROI_keyword='meltpool', self_axis_keyword='principal', 
                   align_axis=np.array([0.,1.]),
                   self_axis_user_def=np.array([None, None]),
                   pixel_index_array_user_def=np.array([[None, None]]),
                   pixel_val_array_user_def=np.array([None])):
        """
        ROI_keyword: 'meltpool' or 'total' or 'other'. 
        self_axis_keyword: 'principal' or 'secondary' or 'other'. 
        """

        straightened_version = copy.deepcopy(self.blank_version())
        
        if not self._isEmpty:
            if self_axis_keyword == 'principal' or self_axis_keyword == 'secondary': 
                self_axis = self._meltpool_axis_of(self_axis_keyword)
            elif self_axis_keyword == 'other':
                self_axis = copy.deepcopy(self_axis_user_def).rehsape(-1)
            else: pass
            
            if ROI_keyword == 'meltpool':
                straightened_version = self._center_rotate_image(self.meltpool_pixel_index_array, 
                                                                 self.meltpool_pixel_intensity_array, 
                                                                 self_axis, align_axis)
                straightened_version = self.refine_asper_threshold(straightened_version, 
                                                                   self.intensity_threshold[0])
                self.meltpool_image_straightened = copy.deepcopy(straightened_version)
                
            elif ROI_keyword == 'total':
                straightened_version = self._center_rotate_image(self._bright_pixel_index_array, 
                                                                 self._bright_pixel_intensity_array, 
                                                                 self_axis, align_axis)
                straightened_version = self.refine_asper_threshold(straightened_version, 
                                                                   self.intensity_threshold[0])
                self.visible_image_straightened = copy.deepcopy(straightened_version)
                
            elif (ROI_keyword == 'other' and pixel_index_array_user_def.all() != None and 
                  pixel_val_array_user_def.all() != None):
                straightened_version = self._center_rotate_image(pixel_index_array_user_def, 
                                                                 pixel_val_array_user_def, 
                                                                 self_axis, align_axis)
                straightened_version = self.refine_asper_threshold(straightened_version, 
                                                                   self.intensity_threshold[0])
                
            else: pass
        
        else: pass

        return straightened_version


    @staticmethod
    def _get_center_of(pixel_index_array):
        """
        """

        return np.mean(pixel_index_array, axis=0).astype(int)


    @staticmethod
    def Hu_moments(image_matrix, feature_ind_list=[]):
        """
        """
        
        hu_moments = moments.HuMoments(image_matrix=image_matrix)
        hu_moments_array = hu_moments.Hu_moments()

        if feature_ind_list != []: 
            return np.array([hu_moments_array[i] for i in feature_ind_list]).reshape(-1)
        else: return hu_moments_array


    @staticmethod
    def Zernike_moments(image_matrix):
        """
        """
        
        pass
    

# class MeltPool(imgBasics.Frame):
class MeltPool(Frame):
    """
    """
    
    def __init__(self, meltpool_file_path=None, meltpool_image_matrix=np.array([None])):
        """
        """
        
        super(MeltPool, self).__init__(file_path=meltpool_file_path, 
                                       image_matrix=meltpool_image_matrix)
        
        self.length = None # Float. 
        self.width = None # Float. 
        
        






def main():
    """
    """

    image_data_directory = os.path.join(DATA_DIRECTORY, IMAGE_DATA_SUBDIR)
    raw_image_folder_list = os.listdir(image_data_directory)

    for raw_image_folder in raw_image_folder_list:
        frame_file_path_temp = glob.glob(os.path.join(image_data_directory, raw_image_folder, 
                                                      "*.{}".format(IMAGE_EXTENSION)))
        frame_temp = Frame(file_path=frame_file_path_temp, intensity_threshold=INTENSITY_THRESHOLD)

        image_straightened_meltpool_temp = frame_temp.straighten('meltpool')
        hu_moments_array_temp = frame_temp.Hu_moments(image_matrix=frame_temp.thresheld_image(), 
                                                      feature_ind_list=HU_MOMENTS_FEATURE_IND_LIST)
        zernike_moments_array_temp = frame_temp.Zernike_moments(image_matrix=frame_temp.thresheld_image())
        
        image_straightened_binarized_meltpool_temp = frame_temp.binarize(image_straightened_meltpool_temp)
        cv2.findContours() # Change a contour extracting method. 




    



if __name__ == "__main__":
    # main()
    
    # Test lines. 
    
    file_path = 'C:/Users/hlinl/Desktop/New folder/data/raw_image_data/Layer012_Section_02_S0001/Layer012_Section_02_S0001001774.png'
    frame = Frame(file_path=file_path, intensity_threshold=INTENSITY_THRESHOLD)

    meltpool_straightened_image = frame.straighten('meltpool', FRAME_ALIGN_MODE, FRAME_REALIGN_AXIS_VECT)
    
    im = frame.binarize(meltpool_straightened_image).astype(np.uint8)
    contours, hierarchy = cv2.findContours(im, mode=cv2.RETR_TREE, 
                                           method=cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im, contours, -1, (0,255,0), 3)
    cv2.imshow('output', im)
    
    i, color, size = 0, 'g', 0.5
    
    plt.figure()
    plt.imshow(frame.image_matrix, cmap='gray')
    
    plt.figure()
    plt.imshow(meltpool_straightened_image, cmap='gray')
    
    plt.figure()
    plt.imshow(im, cmap='gray')
    
    plt.figure()
    plt.imshow(meltpool_straightened_image, cmap='gray')
    plt.scatter(contours[i][:,0,0], contours[i][:,0,1], c=color, s=size)
    
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.scatter(contours[i][:,0,0], contours[i][:,0,1], c=color, s=size)
    
    
    