# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 00:14:04 2021

@author: hlinl
"""


import os
import glob
import copy

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mig

from cmath import nan
from scipy.stats import skew, kurtosis

from PARAM.IMG import *
from PARAM.BASIC import *
from dataprocessing import *


class Frame(imgBasics.Image):
    """
    """

    def __init__(self, file_path=None, image_matrix=np.array([None]), 
                 intensity_threshold=(0.8,1.0)):
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
        self._meltpool_principal_axes = np.array([None]) # Float. 

        self._meltpool_spatial_skewness = None # Float. 
        self._meltpool_intensity_kurtosis = None # Float.
        self._total_intensity_kurtosis = None # Float 

        self.visible_image_aligned = np.array([None])
        self._visible_pixel_index_aligned = np.array([[None, None]])
        self.meltpool_image_aligned = np.array([None])
        self._meltpool_pixel_index_aligned = np.array([[None, None]])

        self._thresholding()
        self._set_meltpool()
        self._set_meltpool_principal_axes()
    

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
    def _rotate(pixel_index_array, theta, center_pt):
        """
        pixel_index_array: shape=(n_points, n_features)
        theta: in degrees. 
        center_pt: shape=(n_features,)
        """

        return imgBasics.rotate_2D(pixel_index_array, theta, center_pt).astype(int)


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

    
    def _set_meltpool_principal_axes(self):
        """
        """

        if not self._isEmpty: 
            pca_coder = pca.PCA(self.meltpool_pixel_index_array.astype(float), 
                                PC_num=PC_NUM_FRAME, mode=PCA_MODE_FRAME)
            self._meltpool_principal_axes = pca_coder.eigFaces()
        else:
            self._meltpool_principal_axes = np.array([[None, None], [None, None]])


    def _compute_meltpool_spatial_skewness(self, axis='principal'):
        """
        axis: 'principal' or 'secondary'. 
        """

        if self._isEmpty: self._meltpool_skewness = None

        else: 
            inner_product_list = []
            for i in range(self.meltpool_pixel_index_array.shape[0]):
                target_vect_temp = copy.deepcopy(self.meltpool_pixel_index_array[i,:]) - self.meltpool_center_pt
                
                if axis == 'secondary': inner_product_temp = target_vect_temp @ self._meltpool_principal_axes[:,1]
                else: inner_product_temp = target_vect_temp @ self._meltpool_principal_axes[:,0]

                inner_product_list.append(inner_product_temp)
                
            self._meltpool_spatial_skewness = skew(inner_product_list)
        
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


    def _realign_index(self, pixel_index_array, align_axis=np.array([0.,1.])):
        """
        Always meltpool center. 
        """

        pixel_index_array_aligned = np.array([[None, None]])

        if not self._isEmpty:
            skewness = self._compute_meltpool_spatial_skewness(axis=FRAME_ALIGN_MODE)

            if skewness <= 0.: axis_vect = copy.deepcopy(self._meltpool_principal_axes[:,0]).reshape(-1)
            else: axis_vect = copy.deepcopy(-self._meltpool_principal_axes[:,0]).reshape(-1)

            theta = utility.angle_2vect(axis_vect, align_axis)
            if axis_vect[0] <= 0.: theta *= -1

            pixel_index_array_rotated = self._rotate(pixel_index_array, theta, self.meltpool_center_pt)
            # center_pt_rotated = self._get_center_of(pixel_index_array_rotated)

            pixel_index_array_aligned = copy.deepcopy(self._centering(pixel_index_array_rotated, self.meltpool_center_pt))

        else: pass

        return pixel_index_array_aligned


    def meltpool_aligned_version(self, keyword='meltpool', 
                                 pixel_index_array_user_def=np.array([[None, None]]),
                                 pixel_val_array_user_def=np.array([None])):
        """
        keyword: 'meltpool' or 'total' or 'other'. 
        """

        aligned_version = copy.deepcopy(self.blank_version())

        if not self._isEmpty:
            if keyword == 'meltpool': 
                self._meltpool_pixel_index_aligned = self._realign_index(self.meltpool_pixel_index_array, 
                                                                         align_axis=FRAME_REALIGN_AXIS_VECT)
                aligned_version = self._specify_image(self.blank_version(), self._meltpool_pixel_index_aligned, 
                                                      self.meltpool_pixel_intensity_array)
                self.meltpool_image_aligned = copy.deepcopy(aligned_version)

            elif keyword == 'total': 
                self._visible_pixel_index_aligned = self._realign_index(self._bright_pixel_index_array, 
                                                                        align_axis=FRAME_REALIGN_AXIS_VECT)
                aligned_version = self._specify_image(self.blank_version(), self._visible_pixel_index_aligned, 
                                                      self._bright_pixel_intensity_array)
                self.visible_image_aligned = copy.deepcopy(aligned_version)

            elif (keyword == 'other' and 
                  pixel_index_array_user_def.all() != None and 
                  pixel_val_array_user_def.all() != None): 
                pixel_index_aligned = self._realign_index(pixel_index_array_user_def, 
                                                          align_axis=FRAME_REALIGN_AXIS_VECT)
                aligned_version = self._specify_image(self.blank_version(), pixel_index_aligned, 
                                                      pixel_val_array_user_def)

            else: pass
        
        else: pass

        return aligned_version


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
    def Zernike_moments(image_matrix, feature_ind_list=[]):
        """
        """
        
        pass
    

def main():
    """
    """

    image_data_directory = os.path.join(DATA_DIRECTORY, IMAGE_DATA_SUBDIR)
    raw_image_folder_list = os.listdir(image_data_directory)

    for raw_image_folder in raw_image_folder_list:
        frame_file_path_temp = glob.glob(os.path.join(image_data_directory, raw_image_folder, 
                                                      "*.{}".format(IMAGE_EXTENSION)))
        frame_temp = Frame(file_path=frame_file_path_temp, intensity_threshold=INTENSITY_THRESHOLD)

        meltpool_aligned_image_temp = frame_temp.meltpool_aligned_version(keyword='meltpool')
        hu_moments_array_temp = frame_temp.Hu_moments(image_matrix=frame_temp.thresheld_image(), 
                                                      feature_ind_list=HU_MOMENTS_FEATURE_IND_LIST)
        zernike_moments_array_temp = frame_temp.Zernike_moments(image_matrix=frame_temp.thresheld_image(), 
                                                                feature_ind_list=HU_MOMENTS_FEATURE_IND_LIST)














    file_path = 'C:/Users/hlinl/OneDrive/Desktop/New folder/Data/raw_image_data/Layer012_Section_02_S0001/Layer012_Section_02_S0001002270.png'
    frame = Frame(file_path=file_path, intensity_threshold=INTENSITY_THRESHOLD)

    meltpool_aligned_image = frame.meltpool_aligned_version(keyword='meltpool')

    print(frame._compute_meltpool_spatial_skewness(axis='principal'))
    print(frame._meltpool_principal_axes)

    plt.figure()
    plt.imshow(meltpool_aligned_image)



if __name__ == "__main__":
    main()