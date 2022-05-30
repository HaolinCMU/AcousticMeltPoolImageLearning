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
        self._set_cluster_dict()
        
    
    def _clustering(self):
        """
        """

        clustering = skc.DBSCAN(eps=self.epsilon, min_samples=self.minPts).fit(self.sample_pts_array)
        self._sample_pts_label_array = clustering.labels_ # Integer labels. 
        self._cluster_label_list = list(set(self._sample_pts_label_array))


    def _set_cluster_dict(self):
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
        
        return largest_cluster_label, self._cluster_dict[largest_cluster_label]

    
    @property
    def noise_cluster(self):
        """
        """

        return self._cluster_dict[-1]

    
    def labels_and_clusters(self, sort='as_is', include_noise=False, exclude=None):
        """
        Return: two lists of labels and clusters, respectively. 
        `exclude`: a list of the labels of clusters to be excluded. 
        """

        # Collect all labels and clusters into separate lists. 
        label_list, cluster_list = [], []
        for key, val in self._cluster_dict.items():
            label_list.append(key)
            cluster_list.append(val)

        if not include_noise and -1 in label_list: # Remove noise(-1) cluster if True. 
            del cluster_list[label_list.index(-1)]
            label_list.remove(-1)
        
        if exclude is not None: # Remove clusters as per the given excluding list. 
            for label_ex in exclude:
                if label_ex in label_list:
                    del cluster_list[label_list.index(label_ex)]
                    label_list.remove(label_ex)
                else: pass

        if sort == 'as_is': return label_list, cluster_list

        elif sort == 'size_descend':
            cluster_size_list = [cluster.shape[0] for cluster in cluster_list]

            zipped = list(zip(cluster_size_list, label_list, cluster_list))
            zipped.sort(key=lambda x: x[0], reverse=True)
            _, label_list, cluster_list = list(zip(*zipped))

            return list(label_list), list(cluster_list)
            
        elif sort == 'size_ascend':
            cluster_size_list = [cluster.shape[0] for cluster in cluster_list]

            zipped = list(zip(cluster_size_list, label_list, cluster_list))
            zipped.sort(key=lambda x: x[0], reverse=False)
            _, label_list, cluster_list = list(zip(*zipped))

            return list(label_list), list(cluster_list)
            
        elif sort == 'asper_label': # Sort as per the order of values of labels (ascend). 
            zipped = list(zip(label_list, cluster_list))
            zipped.sort(key=lambda x: x[0], reverse=False)
            label_list, cluster_list = list(zip(*zipped))

            return list(label_list), list(cluster_list)
        
        else: raise ValueError("Incorrect argument for `sort`. Try using one of the following: \
                                'as_is', 'asper_label', 'size_ascend', 'size_descend'. ")


    @staticmethod
    def center_pt(cluster):
        """
        """

        return np.mean(cluster, axis=0)
    

    @staticmethod
    def size_of(cluster):
        """
        """

        return cluster.shape[0]


class kmeans(object):
    """
    """

    def __init__(self, data_matrix, n_clusters, random_state=0):
        """
        """

        self.data_matrix = data_matrix
        self.n_clusters = n_clusters
        self.random_state = random_state

        self._kmeans = None
        self._labels = None
        self._cluster_centers = None
        self._distance = None

    
    @property
    def labels(self):
        """
        """

        return self._labels
    

    @property
    def cluster_centers(self):
        """
        """

        return self._cluster_centers


    def clustering(self):
        """
        """

        self._kmeans = skc.KMeans(self.n_clusters, random_state=self.random_state).fit(self.data_matrix)
        self._labels = self._kmeans.labels_
        self._cluster_centers = self._kmeans.cluster_centers
        