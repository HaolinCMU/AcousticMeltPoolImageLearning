# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:12:31 2022

@author: hlinl
"""

import os
import copy
import sys
DIR_ABS = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(DIR_ABS))

import numpy as np
import sklearn.cluster as skc

from .imgBasics import *
from .utility import *

from PARAM import *


class dbscan(object):
    """
    """

    def __init__(self, sample_pts_array, epsilon, minPts):
        """
        """

        self.sample_pts_array = sample_pts_array
        self.epsilon = epsilon
        self.minPts = minPts

        self._sample_pts_label_array = None
        self._cluster_label_list = None # 1D array of labels of all clusters. 
        self._cluster_dict = {}

        self._clustering()
        self._generate_cluster_dict()
        
    
    def _clustering(self):
        """
        """

        clustering = skc.DBSCAN(eps=self.epsilon, min_samples=self.minPts).fit(self.sample_pts_array)
        self._sample_pts_label_array = clustering.labels_
        self._cluster_label_list = list(set(self._sample_pts_label_array))


    def _generate_cluster_dict(self):
        """
        """

        for cluster_label in self._cluster_label_list:
            sample_pts_indices_thisCluster = np.where(self._sample_pts_label_array==cluster_label)[0].reshape(-1) 
            sample_pts_thisCluster = self.sample_pts_array[sample_pts_indices_thisCluster,:]
            self._cluster_dict[cluster_label] = sample_pts_thisCluster
            

    def largest_cluster(self):
        """
        """

        largest_cluster_label, largest_cluster_size = -1, 0

        for key, val in self._cluster_dict.items():
            if key == -1: continue # Skip the noise cluster.

            if val.shape[0] >= largest_cluster_size: 
                largest_cluster_label = key
                largest_cluster_size = val.shape[0]
            else: continue
        
        return self._cluster_dict[largest_cluster_label]


    @staticmethod
    def center_pt(cluster):
        """
        """

        return np.mean(cluster, axis=0)


class kmean(object):
    """
    """

    def __init__(self):
        """
        """

        pass