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
import PIL

from cmath import nan
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import skew, kurtosis
from scipy.ndimage import rotate
from torchvision import transforms

from PARAM.IMG import *
from PARAM.BASIC import *
from PARAM.ACOUSTIC import *
from dataprocessing import *
from files import *


class Frame(imgBasics.Image):
    """
    """

    def __init__(self, file_path=None, image_matrix=np.array([None]), intensity_threshold=INTENSITY_THRESHOLD, 
                 sidebar_threshold=SIDEBAR_THRESHOLD, sidebar_columns=SIDEBAR_COLUMNS, plume_threshold=PLUME_THRESHOLD):
        """
        """

        super(Frame, self).__init__(file_path, image_matrix)
        self.intensity_threshold = intensity_threshold # Tuple of Floats. Two constant intensity thresholds that filter out the bright melt pool and spatters. 
        self.sidebar_threshold = sidebar_threshold # Float. A constant intensity threshold for sidebar filtering. Default: 0.05. 
        self.sidebar_columns = sidebar_columns # List of tuples. Two tuples of int that prescribe the range of side bar. 
        self.plume_threshold = plume_threshold # Tuple of Floats. Two constant intensity thresholds that filter out the plumes. 
        self._isEmpty = False # Boolean. True if no pixels satisfy the intensity threshold. 
        
        # Original. 
        self.original_image_matrix = copy.deepcopy(self.image_matrix) # 2D array of Float. (512, 512). 0.-1. Save a copy of original unprocessed image matrix. 
        self.original_pixel_index_array = None
        self.original_pixel_intensity_array = None
        
        # Total. 
        self._bright_pixel_index_array = np.array([[None, None]])
        self._bright_pixel_intensity_array = np.array([None]) # Same order as `self._bright_pixel_index_array`. 
        self.visible_image = np.array([None])
        self.mass_center_pt = np.array([None]) # 1D Int array. A pixel. 
        self._total_area = None

        self._pixel_clusters = None

        # Melt pool. 
        self._isMeltpool = False
        self.meltpool_pixel_index_array = np.array([[None, None]])
        self.meltpool_pixel_intensity_array = np.array([None]) # Same order as `self.meltpool_pixel_intensity_array`.
        self._meltpool_image = np.array([None])
        self.meltpool_center_pt = np.array([None]) # 1D Int array. A pixel. 
        self._meltpool_area = None # Int. Number of melt pool pixels. 
        self._meltpool_length = None # Float. 
        self._meltpool_width = None # Float. 
        self._meltpool_principal_axes = None # Float. 

        self._meltpool_spatial_skewness = None # Float. 
        self._spatters_intensity_kurtosis = None # Float. 
        self._meltpool_intensity_kurtosis = None # Float.
        self._total_intensity_kurtosis = None # Float. 

        # Spatters. 
        self._isSpatter = False
        self.spatters_list = None
        self.spatters_pixel_index_array = None
        self.spatters_pixel_intensity_array = None
        self._spatters_image = None
        self.spatters_center_pts = None # List of center points. 
        self._spatters_num = 0
        self._spatters_sizes = None
        self._spatters_distances = None
        
        # Plume. 
        self._isPlume = False
        self.plume_pixel_index_array = None
        self.plume_pixel_intensity_array = None
        self._plume_image = None
        self._plume_area = None
        self._plume_convex_hull = None # Not employed. 
        
        # Straightened results. 
        self.visible_image_straightened = None
        self.meltpool_image_straightened = None 
        self.spatters_image_straightened = None
        self.plume_image_straightened = None
        
        self._preprocess() # Frame image preprocessing. 
        self._thresholding() # Image thresholding. 
        self._set_meltpool() # Define meltpool. 
        self._set_spatters() # Define spatters. 
        self._set_plume() # Define plume. 
    

    @property
    def total_area(self):
        """
        """

        return self._total_area


    @property
    def meltpool_aspect_ratio(self):
        """
        """

        if not self._isMeltpool or self._meltpool_width == 0: return 0.
        else: return self._meltpool_length / self._meltpool_width


    @property
    def meltpool_area(self):
        """
        """

        return self._meltpool_area
    
    
    @property
    def plume_area(self):
        """
        """

        return self._plume_area


    @property
    def spatters_sizes(self):
        """
        """

        return self._spatters_sizes
    

    @property
    def spatters_num(self):
        """
        """

        return self._spatters_num


    @property
    def visible_pixel_indices(self):
        """
        Get the indices of pixels with intensity values fallen in between the range defined by `self.intensity_threshold`. 
        """ 

        return self._bright_pixel_index_array


    @property
    def meltpool_pixel_indices(self):
        """
        Get the indices of pixels with intensity values fallen in between the range defined by `self.intensity_threshold`. 
        """ 

        return self.meltpool_pixel_index_array
    

    @property
    def spatters_pixel_indices(self):
        """
        """

        return self.spatters_pixel_index_array
    
    
    @property
    def frame(self):
        """
        Return the original, unprocessed image matrix. 
        """
        
        return self.original_image_matrix


    @property
    def thresheld_image(self):
        """
        """
        
        return self.visible_image

    
    @property
    def meltpool_image(self):
        """
        """
        
        return self._meltpool_image
    

    @property
    def spatters_image(self):
        """
        """

        return self._spatters_image
    
    
    @property
    def plume_image(self):
        """
        """

        return self._plume_image
    
    
    @property
    def meltpool_length(self):
        """
        """
        
        return self._meltpool_length
    
    
    @property
    def meltpool_width(self):
        """
        """
        
        return self._meltpool_width

    
    @property
    def spatters_distances(self):
        """
        """

        return self._spatters_distances
    

    @staticmethod
    def _dimension_along_axis(pixel_index_array, origin_pt, axis_vect):
        """
        `pixel_index_array`: shape=(n_sample, n_feature). 
        `origin_pt`: shape=(n_feature,). 
        `axis_vect`: shape=(n_feature,). 
        """
        
        inner_product_array = imgBasics.projection_along_axis(pixel_index_array, origin_pt, axis_vect)
        
        return max(inner_product_array) - min(inner_product_array)
    
    
    def _preprocess(self):
        """
        Remove background noise locating at the two visible bright bars on both sides of the image frame. 
        """
        
        preprocessed_matrix = copy.deepcopy(self.original_image_matrix)
        
        for sidebar_col_range in self.sidebar_columns:
            preprocessed_matrix[:,sidebar_col_range[0]:sidebar_col_range[1]+1]\
                               [np.where(preprocessed_matrix[:,sidebar_col_range[0]:sidebar_col_range[1]+1]\
                                         <self.sidebar_threshold)] = self._default_background_value
        
        self.image_matrix = copy.deepcopy(preprocessed_matrix)
        
        # Save pixels and intensity values of the original image. 
        original_indices_array_row_col = np.where(np.logical_and(self.image_matrix >= 0., self.image_matrix <= 1.))
        self.original_pixel_index_array = np.vstack((original_indices_array_row_col[0], 
                                                     original_indices_array_row_col[1])).astype(int).T # [row | col]; [Axis-0 | Axis-1].
        self.original_pixel_intensity_array = copy.deepcopy(self.image_matrix.reshape(-1))
        
    
    def _set_meltpool_dimensions(self):
        """
        """
        
        if not self._isEmpty and self._isMeltpool:
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
 
        image_dim_0, image_dim_1 = image.shape
        image_specified = copy.deepcopy(image)

        for i in range(pixel_index_array.shape[0]):
            row_ind_temp, col_ind_temp = pixel_index_array[i,:].astype(int)

            if (row_ind_temp >= 0 and
                row_ind_temp < image_dim_0 and
                col_ind_temp >= 0 and
                col_ind_temp < image_dim_1): # Make sure that the indices is within the new image's frame range. Discard the outliers. 
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


    def binarize(self, image_matrix, threshold=None):
        """
        Binarize image by filtering out all background pixels. 
        """

        if threshold is None: return (image_matrix>self._default_background_value).astype(float)
        else: return (image_matrix>threshold).astype(float)


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
        Use thresholds to filter different regions of frame of interest. 
        """
        
        # Bright pixels. 
        bright_indices_array_row_col = np.where(np.logical_and(self.image_matrix >= self.intensity_threshold[0], 
                                                               self.image_matrix <= self.intensity_threshold[1]))
        self._bright_pixel_index_array = np.vstack((bright_indices_array_row_col[0], 
                                                    bright_indices_array_row_col[1])).astype(int).T # [row | col]; [Axis-0 | Axis-1].
        self._total_area = self._bright_pixel_index_array.shape[0]

        if self._total_area != 0:
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
            
        # Plume pixels. 
        plume_indices_array_row_col = np.where(np.logical_and(self.image_matrix >= self.plume_threshold[0], 
                                                              self.image_matrix <= self.plume_threshold[1]))
        self.plume_pixel_index_array = np.vstack((plume_indices_array_row_col[0], 
                                                  plume_indices_array_row_col[1])).astype(int).T # [row | col]; [Axis-0 | Axis-1].
        self._plume_area = self.plume_pixel_index_array.shape[0]


    def _set_meltpool(self):
        """
        Extract the pixel indices of melt pool, which is the largest pixel cluster in the filtered image. 
        """

        self.meltpool_pixel_index_array = np.array([[None, None]])
        self.meltpool_pixel_intensity_array = np.array([None])
        self._meltpool_image = self.blank_version()
        self.meltpool_center_pt = copy.deepcopy(self.mass_center_pt)
        self._meltpool_area = 0

        if not self._isEmpty:
            self._pixel_clusters = clustering.dbscan(self._bright_pixel_index_array, DBSCAN_EPSILON, DBSCAN_MIN_PTS)
            meltpool_cluster_label, meltpool_cluster = self._pixel_clusters.largest_cluster()

            if meltpool_cluster_label != -1: 
                self._isMeltpool = True
                self.meltpool_pixel_index_array = meltpool_cluster.astype(int) # Discard the label of the largest cluster. 
                self.meltpool_pixel_intensity_array = copy.deepcopy(np.array([self.image_matrix[i,j] for i,j in \
                                                                              self.meltpool_pixel_index_array]).reshape(-1))
                self._meltpool_image = self._filter_image(self.meltpool_pixel_index_array)
                self.meltpool_center_pt = self._pixel_clusters.center_pt(self.meltpool_pixel_index_array).astype(int)
                self._meltpool_area = self._pixel_clusters.size_of(self.meltpool_pixel_index_array)
            
            else: self._isMeltpool = False

        else: pass
        
        self._set_meltpool_principal_axes()
        self._set_meltpool_dimensions()
        self._compute_meltpool_spatial_skewness(axis=self._meltpool_axis_of('principal')) # Default: spatial skewness along 'principal' axis. 
        self._compute_intensity_kurtosis() # Default: intensity kurtosis of 'meltpool'. 

    
    def _set_meltpool_principal_axes(self):
        """
        """

        if not self._isEmpty and self._isMeltpool: 
            pca_coder = pca.PCA(self.meltpool_pixel_index_array.astype(float), 
                                PC_num=PC_NUM_FRAME, mode=PCA_MODE_FRAME)
            self._meltpool_principal_axes = pca_coder.eigFaces()
        else:
            self._meltpool_principal_axes = np.array([[0., 1.], [1., 0.]])

        
    def _meltpool_axis_of(self, keyword='principal'):
        """
        """
        
        meltpool_axis = np.array([0., 1.])
        
        if not self._isEmpty and self._isMeltpool:
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

        if not self._isEmpty and self._isMeltpool:             
            inner_product_array = imgBasics.projection_along_axis(self.meltpool_pixel_index_array, 
                                                                  self.meltpool_center_pt, axis)
            self._meltpool_spatial_skewness = skew(inner_product_array)
        
        else: self._meltpool_spatial_skewness = None
        
        return self._meltpool_spatial_skewness

    
    def _compute_intensity_kurtosis(self, keyword='meltpool'): 
        """
        """

        kurtosis_val = None

        if keyword == 'meltpool':
            if not self._isEmpty and self._isMeltpool: 
                kurtosis_val = kurtosis(self.meltpool_pixel_intensity_array)
            else: pass

            self._meltpool_intensity_kurtosis = kurtosis_val

        elif keyword == 'total':
            if not self._isEmpty: 
                kurtosis_val = kurtosis(self._bright_pixel_intensity_array)
            else: pass

            self._total_intensity_kurtosis = kurtosis_val

        elif keyword == 'spatters':
            if not self._isEmpty and self._isSpatter: 
                kurtosis_val = kurtosis(self.spatters_pixel_intensity_array)
            else: pass

            self._spatters_intensity_kurtosis = kurtosis_val

        else: pass
    
        return kurtosis_val
    

    def _set_spatters(self):
        """
        """
        
        self.spatters_pixel_index_array = np.array([[None, None]])
        self.spatters_pixel_intensity_array = np.array([None])
        self._spatters_image = self.blank_version()
        self.spatters_center_pts = [copy.deepcopy(self.mass_center_pt)]

        if not self._isEmpty and self._isMeltpool and self._pixel_clusters is not None:
            _, cluster_list = self._pixel_clusters.labels_and_clusters(sort='size_descend')
            self.spatters_list = copy.deepcopy(cluster_list[1:]) # Exclude the largest cluster (melt pool). 
            self._spatters_num = len(self.spatters_list)

            if self._spatters_num != 0: 
                self._isSpatter = True
                self.spatters_center_pts = [self._pixel_clusters.center_pt(spatter_cluster) \
                                            for spatter_cluster in self.spatters_list]
                self._spatters_sizes = [self._pixel_clusters.size_of(spatter_cluster) \
                                        for spatter_cluster in self.spatters_list]
                self._spatters_distances = [np.linalg.norm(spatter_center_pt-self.meltpool_center_pt) \
                                            for spatter_center_pt in self.spatters_center_pts]
            
                self.spatters_pixel_index_array = utility.A_diff_B_array(A=self._bright_pixel_index_array, 
                                                                         B=self.meltpool_pixel_index_array).astype(int)
                self.spatters_pixel_intensity_array = copy.deepcopy(np.array([self.image_matrix[i,j] \
                                                                    for i,j in self.spatters_pixel_index_array]).reshape(-1))
                self._spatters_image = self._filter_image(self.spatters_pixel_index_array)
            
            else: self._isSpatter = False
            
        else: pass
    
    
    def _set_plume(self):
        """
        """
        
        self._isPlume = False
        self.plume_pixel_intensity_array = None
        self._plume_image = self.blank_version()
        
        if not self._isEmpty and self._isMeltpool:
            if self._plume_area != 0:
                self._isPlume = True
                self.plume_pixel_intensity_array = copy.deepcopy(np.array([self.image_matrix[i,j] 
                                                                           for i,j in self.plume_pixel_index_array]).reshape(-1))
                self._plume_image = self._filter_image(self.plume_pixel_index_array)
            
            else: pass
        
        else: pass
    
   
    def _set_straighten_angle(self, meltpool_axis, align_axis=np.array([0.,1.])):
        """
        """
        
        rotation_angle = 0.
        
        if not self._isEmpty and self._isMeltpool:
            skewness = self._compute_meltpool_spatial_skewness(meltpool_axis)
            
            if skewness <= 0.: axis_vect = copy.deepcopy(meltpool_axis)
            else: axis_vect = -copy.deepcopy(meltpool_axis)

            rotation_angle = utility.angle_2vect(axis_vect, align_axis)
            
            if np.sign(np.cross(align_axis, axis_vect)) >= 0.: rotation_angle *= -1
            else: pass

        else: pass
        
        return rotation_angle
    

    def _center_rotate_image(self, pixel_index_array, pixel_val_array, center_pt, 
                             self_axis, align_axis=np.array([0.,1.])):
        """
        Always meltpool center. 
        """
        
        pixel_index_array_centered = self._centering(pixel_index_array, center_pt)
        centered_version = self._specify_image(self.blank_version(), pixel_index_array_centered, pixel_val_array)
        
        theta = self._set_straighten_angle(meltpool_axis=self_axis, align_axis=align_axis)
        straightened_version = copy.deepcopy(self._rotate(centered_version, theta))
        
        return straightened_version


    def straighten(self, ROI_keyword='meltpool', self_axis_keyword='principal', align_axis=np.array([0.,1.]), 
                   self_axis_user_def=None, pixel_index_array_user_def=None, pixel_val_array_user_def=None, 
                   pixel_center_array_user_def=None):
        """
        ROI_keyword: 'meltpool' or 'total' or 'spatter' or 'plume' or 'original' or 'other'. 
        self_axis_keyword: 'principal' or 'secondary' or 'other'. 
        """

        straightened_version = copy.deepcopy(self.blank_version())
        
        if not self._isEmpty:
            if self_axis_keyword == 'principal' or self_axis_keyword == 'secondary': 
                self_axis = self._meltpool_axis_of(self_axis_keyword)
            elif self_axis_keyword == 'other':
                self_axis = copy.deepcopy(self_axis_user_def).reshape(-1)
            else: pass
            
            if ROI_keyword == 'original':
                straightened_version = self._center_rotate_image(self.original_pixel_index_array, 
                                                                 self.original_pixel_intensity_array, 
                                                                 self.meltpool_center_pt, self_axis, align_axis) # Can change center point of meltpool or mass (total). 
                self.meltpool_image_straightened = copy.deepcopy(straightened_version)
            
            elif ROI_keyword == 'meltpool' and self._isMeltpool:
                straightened_version = self._center_rotate_image(self.meltpool_pixel_index_array, 
                                                                 self.meltpool_pixel_intensity_array, 
                                                                 self.meltpool_center_pt, self_axis, align_axis)
                straightened_version = self.refine_asper_threshold(straightened_version, 
                                                                   self.intensity_threshold[0])
                self.meltpool_image_straightened = copy.deepcopy(straightened_version)
                
            elif ROI_keyword == 'total':
                straightened_version = self._center_rotate_image(self._bright_pixel_index_array, 
                                                                 self._bright_pixel_intensity_array, 
                                                                 self.meltpool_center_pt, self_axis, align_axis) # Can change center point of meltpool or mass (total). 
                straightened_version = self.refine_asper_threshold(straightened_version, 
                                                                   self.intensity_threshold[0])
                self.visible_image_straightened = copy.deepcopy(straightened_version)
            
            elif ROI_keyword == 'spatters' and self._isSpatter:
                straightened_version = self._center_rotate_image(self.spatters_pixel_index_array, 
                                                                 self.spatters_pixel_intensity_array, 
                                                                 self.meltpool_center_pt, self_axis, align_axis) # Can change center point of meltpool or mass. 
                straightened_version = self.refine_asper_threshold(straightened_version, 
                                                                   self.intensity_threshold[0])
                self.spatters_image_straightened = copy.deepcopy(straightened_version)
            
            elif ROI_keyword == 'plume' and self._isPlume: 
                straightened_version = self._center_rotate_image(self.plume_pixel_index_array, 
                                                                 self.plume_pixel_intensity_array, 
                                                                 self.meltpool_center_pt, self_axis, align_axis) # Can change center point of meltpool or mass. 
                straightened_version = self.refine_asper_threshold(straightened_version, 
                                                                   self.plume_threshold[0])
                self.plume_image_straightened = copy.deepcopy(straightened_version)
            
            elif (ROI_keyword == 'other' and pixel_index_array_user_def is not None and 
                  pixel_val_array_user_def is not None and pixel_center_array_user_def is not None):
                straightened_version = self._center_rotate_image(pixel_index_array_user_def, 
                                                                 pixel_val_array_user_def, 
                                                                 pixel_center_array_user_def, self_axis, align_axis)
                straightened_version = self.refine_asper_threshold(straightened_version, 
                                                                   self.intensity_threshold[0])
                
            else: pass
        
        else: pass

        return copy.deepcopy(straightened_version)


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


def collect_visual_data(img_dir, visual_dir, featurization_mode='median',
                        is_area=True, is_aspect_ratio=True, is_P=True, is_V=True): # Add arguments for feature types. 
    """
    Current feature (exact order): [ Area | Aspect ratio | P | V ]. 
    Collect as much as possible. 
    """

    img_subfolder_list = os.listdir(img_dir)

    for folder in img_subfolder_list:
        img_subfolder_dir = os.path.join(img_dir, folder)
        img_path_list_temp = glob.glob(os.path.join(img_subfolder_dir, "*.{}".format(IMAGE_EXTENSION)))

        P_temp, V_temp = extract_process_param_fromAcousticFilename(folder)

        if is_area: area_list = copy.deepcopy([])
        if is_aspect_ratio: aspect_ratio_list = copy.deepcopy([])

        print("Start processing File: {}. ".format(folder))
        print("------------------------------")

        for ind, img_path in enumerate(img_path_list_temp):
            img_frame_temp = Frame(img_path)
            if is_area: area_list.append(img_frame_temp.meltpool_area)
            if is_aspect_ratio: aspect_ratio_list.append(img_frame_temp.meltpool_aspect_ratio)

            if (ind + 1) % 1000 == 0 or ind + 1 == len(img_path_list_temp): 
                print("Img: {} | {}/{} is being processed. ".format(folder, ind+1, len(img_path_list_temp)))

            del img_frame_temp # Release memory. 

        visual_data_mat_full = np.array([]).reshape(len(img_path_list_temp), 0)
        if is_area: 
            area_array = np.array(area_list).astype(float).reshape(-1,1)
            visual_data_mat_full = np.hstack((visual_data_mat_full, area_array))
        if is_aspect_ratio:
            aspect_ratio_array = np.array(aspect_ratio_list).astype(float).reshape(-1,1)
            visual_data_mat_full = np.hstack((visual_data_mat_full, aspect_ratio_array))
        if is_P:
            P_array = np.array([P_temp]*len(img_path_list_temp)).astype(float).reshape(-1,1)
            visual_data_mat_full = np.hstack((visual_data_mat_full, P_array))
        if is_V:
            V_array = np.array([V_temp]*len(img_path_list_temp)).astype(float).reshape(-1,1)
            visual_data_mat_full = np.hstack((visual_data_mat_full, V_array))

        ###: Sliding window.
        visual_data_block = sliding_window_view(visual_data_mat_full, # Shape: (sample_num, 1, IMAGE_WINDOW_SIZE, feature_num). 
                                                window_shape=(IMAGE_WINDOW_SIZE, 
                                                              visual_data_mat_full.shape[1]))[::IMAGE_STRIDE_SIZE,:,:,:]
        if featurization_mode == 'mean': 
            visual_data_mat = np.mean(visual_data_block, axis=2).reshape(-1, visual_data_mat_full.shape[1]) # Mean value. 
        elif featurization_mode == 'median':
            visual_data_mat = np.median(visual_data_block, axis=2).reshape(-1, visual_data_mat_full.shape[1]) # Median value. 
        else: raise ValueError("Featurization mode not interpretable. ")

        visual_subfolder_dir = os.path.join(visual_dir, folder)
        if not os.path.isdir(visual_subfolder_dir): os.mkdir(visual_subfolder_dir)

        for i in range(visual_data_mat.shape[0]):
            visual_data_path_temp = os.path.join(visual_subfolder_dir, "{}_{}.{}".format(folder, str(i).zfill(5), 
                                                                                         IMG.VISUAL_DATA_EXTENSION))
            np.save(visual_data_path_temp, visual_data_mat[i,:])
        
        print("End of processing File: {}. ".format(folder))
        print("##############################")


def visual_data_standard(visual_dir):
    """
    """

    visual_data_subfolder_list = os.listdir(visual_dir)
    visual_data_pathlist_total, visual_data_mat_total = [], None

    for subfolder_ind, visual_data_subfolder in enumerate(visual_data_subfolder_list):
        visual_data_pathlist_temp = glob.glob(os.path.join(visual_dir, visual_data_subfolder, 
                                                           "*.{}".format(IMG.VISUAL_DATA_EXTENSION)))

        for path_ind, visual_data_path in enumerate(visual_data_pathlist_temp):
            visual_data_pathlist_total.append(visual_data_path)
            visual_data_temp = np.load(visual_data_path).reshape(1, -1)

            if subfolder_ind == 0 and path_ind == 0: visual_data_mat_total = copy.deepcopy(visual_data_temp)
            else: visual_data_mat_total = np.vstack((visual_data_mat_total, visual_data_temp))
    
    visual_mean, visual_std = np.mean(visual_data_mat_total, axis=0), np.std(visual_data_mat_total, axis=0)
    visual_data_mat_standard = utility.standardization(visual_data_mat_total, axis=0, 
                                                       mean_vect=visual_mean, std_vect=visual_std)

    if len(visual_data_pathlist_total) != visual_data_mat_standard.shape[0]: raise ValueError("Implementation error. ")
    else: visual_data_num_total = len(visual_data_pathlist_total)

    for i in range(visual_data_num_total): np.save(visual_data_pathlist_total[i], visual_data_mat_standard[i,:])

    return visual_mean, visual_std


def main():
    """
    """

    # Set parser.
    parser = argparse.ArgumentParser(description="__init__")

    # Dataset & Data generation. 
    parser.add_argument("--data_dir_path", default=DATA_DIRECTORY, type=str, 
                        help="The directory of all raw data, including audio and image. ")
    parser.add_argument("--img_data_subdir", default=IMAGE_DATA_SUBDIR, type=str, 
                        help="The folder name of raw image data, including subfolders of different layers. ")
    parser.add_argument("--img_processed_data_subdir", default=IMAGE_PROCESSED_DATA_SUBDIR, type=str, 
                        help="The folder name of processed image data, including subfolders of different layers. ")
    parser.add_argument("--img_extension", default=IMAGE_EXTENSION, type=str, 
                        help="The extension (file format) of raw image files. ")
    parser.add_argument("--image_size", default=IMAGE_SIZE, type=list, 
                        help="The intended size ([h, w]) of the processed image. ")
    parser.add_argument("--img_straighten_keyword", default=IMG_STRAIGHTEN_KEYWORD, type=str, 
                        help="The keyword of image straighten mode for the class `Frame`. ")
    parser.add_argument("--frame_align_mode", default=FRAME_ALIGN_MODE, type=str, 
                        help="The keyword indicating the moving axis of the melt pool image frame. ")
    parser.add_argument("--frame_realign_axis_vect", default=FRAME_REALIGN_AXIS_VECT, type=str, 
                        help="The keyword indicating the targeted axis of the melt pool image frame. ")
    parser.add_argument("--is_binary", default=True, type=bool, 
                        help="Indicate whether the processed images require binarization. ")

    args = parser.parse_args()

    # Generate and save straightened melt pool images.  
    img_data_subdir_path = os.path.join(args.data_dir_path, args.img_data_subdir) # Directory of raw melt pool images. 
    img_processed_data_subdir_path = os.path.join(args.data_dir_path, args.img_processed_data_subdir) # Directory of processed melt pool images. 
    clr_dir(img_processed_data_subdir_path)

    img_subfolder_list = os.listdir(img_data_subdir_path) # List of subfolders of different layers.

    for img_subfolder in img_subfolder_list:
        img_processed_subfolder_path = os.path.join(img_processed_data_subdir_path, img_subfolder)
        if not os.path.isdir(img_processed_subfolder_path): os.mkdir(img_processed_subfolder_path) # Create folder for processed images. 

        img_subfolder_path = os.path.join(img_data_subdir_path, img_subfolder)
        img_filepath_perSubFolder_list = glob.glob(os.path.join(img_subfolder_path, "*.{}".format(args.img_extension))) # List of image file paths of each layer's subfolder. 

        for img_ind, img_filepath in enumerate(img_filepath_perSubFolder_list): # Image data processing part can be separated from the main workflow. 
            frame_temp = Frame(img_filepath)
            meltpool_straightened_image_temp = frame_temp.straighten(args.img_straighten_keyword, 
                                                                     args.frame_align_mode,
                                                                     args.frame_realign_axis_vect) # Get straightened meltpool image.

            img_processed_temp = copy.deepcopy(meltpool_straightened_image_temp)
            if not args.is_binary: img_processed_temp = PIL.Image.fromarray(np.uint8(img_processed_temp*255)) # Not binarizing the image, keeping the original intensity values. 
            else: img_processed_temp = PIL.Image.fromarray(np.uint8(frame_temp.binarize(img_processed_temp)*255)) # Binarize image and convert it to `uint8` data type. 
            img_processed_temp = transforms.Resize(args.image_size)(img_processed_temp)

            img_processed_file_path_temp = os.path.join(img_processed_subfolder_path, 
                                                        "{}_{}.{}".format(img_subfolder, img_ind, args.img_extension))
            img_processed_temp.save(img_processed_file_path_temp)


if __name__ == "__main__":
    main()

    # A new way of implementation: 
    # Use a large range of intensity threshold and filter out most of the pixels, then apply another intensity threshold \
    #   on the biggest cluster to separate melt pool, track, plume and additional spatters. 
    # Can we use Kalman filter to track the melt pool more accurately? 
    # Frequently use @property method to define protected attributes, and else: pass to define function, and is None as conditions. 

