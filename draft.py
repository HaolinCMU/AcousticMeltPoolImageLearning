# -*- coding: utf-8 -*-
"""
Created on Wed May 25 04:12:00 2022

@author: hlinl
"""


import os
import glob
import copy
from unittest import result

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
import PIL
import shutil
import sklearn.cluster as skc
import sklearn.metrics as skm
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import manifold

from PARAM import *
from dataprocessing import *
from model import *

from files import *
from image_processing import *
from training import *
from dataset import *


def kmeans_clustering(data_matrix, n_clusters=2):
    """
    method: "DBSCAN" or "KMEANS". 
    n_clusters: only for k-means. 
    file_paths_list: only for DBSCAN. 
    """
    
    kmeans = skc.KMeans(n_clusters, random_state=0).fit(data_matrix)
    return kmeans


def tSNE_plot_2D(data, label_array, figure_name="tSNE_plot.png"):
    """
    """
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(data)
    
    colors_map_list = ['red', 'green', 'blue', 'orange', 'purple', 
                       'pink', 'gray', 'cyan', 'brown', 'olive']
    color_plot_list = []
    for i in range(data.shape[0]):
        color_plot_list.append(colors_map_list[label_array[i]%len(colors_map_list)])
    
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    plt.scatter(Y[:,0], Y[:,1], c=color_plot_list, cmap=plt.cm.Spectral, linewidths=1.0)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(figure_name)


if __name__ == "__main__":
    result_directory = "result/4th_bottleneck=32"
    figure_directory = os.path.join(result_directory, "figures")
    cluster_directory = os.path.join(result_directory, "clusters")

    if not os.path.isdir(figure_directory): os.mkdir(figure_directory)
    if not os.path.isdir(cluster_directory): os.mkdir(cluster_directory)

    input_data_repo_dict_path = os.path.join(result_directory, "input_data_repo_dict.mat")
    output_data_repo_dict_path = os.path.join(result_directory, "output_data_repo_dict.mat")

    groundtruths_list_test_path = os.path.join(result_directory, "groundtruths_list_test.npy")
    generations_list_test_path = os.path.join(result_directory, "generations_list_test.npy")
    groundtruths_list_train_path = os.path.join(result_directory, "groundtruths_list_train.npy")
    generations_list_train_path = os.path.join(result_directory, "generations_list_train.npy")

    latent_list_test_path = os.path.join(result_directory, "latent_list_test.npy")
    mu_list_test_path = os.path.join(result_directory, "mu_list_test.npy")
    latent_list_train_path = os.path.join(result_directory, "latent_list_train.npy")
    mu_list_train_path = os.path.join(result_directory, "mu_list_train.npy")

    test_set_ind_array_path = os.path.join(result_directory, "test_set_ind_array.npy")
    train_set_ind_array_path = os.path.join(result_directory, "train_set_ind_array.npy")
    valid_set_ind_array_path = os.path.join(result_directory, "valid_set_ind_array.npy")

    # ===

    gt_list_test = np.load(groundtruths_list_test_path)
    gr_list_test = np.load(generations_list_test_path)
    gt_list_train = np.load(groundtruths_list_train_path)
    gr_list_train = np.load(generations_list_train_path)

    latent_list_test = np.load(latent_list_test_path)
    mu_list_test = np.load(mu_list_test_path)
    latent_list_train = np.load(latent_list_train_path)
    mu_list_train = np.load(mu_list_train_path)
    
    # ======= Special case, remove after debugging =======
    # latent_list_train = np.load(latent_list_train_path, allow_pickle=True)
    # for j, arr in enumerate(latent_list_train):
    #     for i in range(arr.shape[0]):
    #         if j == 0 and i == 0: temp = arr[i,:].reshape(1,-1)
    #         else: temp = np.vstack((temp, arr[i,:].reshape(1,-1)))
    # latent_list_train = copy.deepcopy(temp)
    # np.save(latent_list_train_path, latent_list_train)

    # mu_list_train = np.load(mu_list_train_path, allow_pickle=True)
    # for j, arr in enumerate(mu_list_train):
    #     for i in range(arr.shape[0]):
    #         if j == 0 and i == 0: temp = arr[i,:].reshape(1,-1)
    #         else: temp = np.vstack((temp, arr[i,:].reshape(1,-1)))
    # mu_list_train = copy.deepcopy(temp)
    # np.save(mu_list_train_path, mu_list_train)
    # ====================================================

    test_set_ind = np.load(test_set_ind_array_path)
    train_set_ind = np.load(train_set_ind_array_path)
    valid_set_ind = np.load(valid_set_ind_array_path)

    input_data_repo_dict = scipy.io.loadmat(input_data_repo_dict_path)
    output_data_repo_dict = scipy.io.loadmat(output_data_repo_dict_path)

    # ===

    n_clusters = 10

    # ===
    # Test set clustering & t-SNE. 
    latent_test_data = latent_list_test.reshape(latent_list_test.shape[0], -1)
    kmeans_latent_test = kmeans_clustering(latent_test_data, n_clusters)
    kmeans_label_array_latent_test, kmeans_center_array_latent_test = kmeans_latent_test.labels_, kmeans_latent_test.cluster_centers_
    tSNE_plot_2D(latent_test_data, kmeans_label_array_latent_test, 
                 figure_name=os.path.join(figure_directory, "latent_test_tsne_c={}.png".format(n_clusters)))
    
    mu_test_data = mu_list_test.reshape(mu_list_test.shape[0], -1)
    kmeans_mu_test = kmeans_clustering(mu_test_data, n_clusters)
    kmeans_label_array_mu_test, kmeans_center_array_mu_test = kmeans_mu_test.labels_, kmeans_mu_test.cluster_centers_
    tSNE_plot_2D(mu_test_data, kmeans_label_array_mu_test, 
                 figure_name=os.path.join(figure_directory, "mu_test_tsne_c={}.png".format(n_clusters)))

    # Copy files only for test dataset. 
    clr_dir(cluster_directory)
    for i in range(n_clusters): 
        subfolder_path_temp = os.path.join(cluster_directory, "{}".format(i))
        if not os.path.isdir(subfolder_path_temp): os.mkdir(subfolder_path_temp)

        cluster_temp = test_set_ind[np.where(kmeans_label_array_mu_test==i)]

        for ind in cluster_temp:
            shutil.copy(output_data_repo_dict[str(ind)][2], subfolder_path_temp+'/')


    # Training set clustering & t-SNE. 
    latent_train_data = latent_list_train.reshape(latent_list_train.shape[0], -1)
    kmeans_latent_train = kmeans_clustering(latent_train_data, n_clusters)
    kmeans_label_array_latent_train, kmeans_center_array_latent_train = kmeans_latent_train.labels_, kmeans_latent_train.cluster_centers_
    tSNE_plot_2D(latent_train_data, kmeans_label_array_latent_train, 
                 figure_name=os.path.join(figure_directory, "latent_train_tsne_c={}.png".format(n_clusters)))
    
    mu_train_data = mu_list_train.reshape(mu_list_train.shape[0], -1)
    kmeans_mu_train = kmeans_clustering(mu_train_data, n_clusters)
    kmeans_label_array_mu_train, kmeans_center_array_mu_train = kmeans_mu_train.labels_, kmeans_mu_train.cluster_centers_
    tSNE_plot_2D(mu_train_data, kmeans_label_array_mu_train, 
                 figure_name=os.path.join(figure_directory, "mu_train_tsne_c={}.png".format(n_clusters)))
    

    # Reconstructed images. 
    for i in range(9):
        gt_test_temp = gt_list_test[i]
        gr_test_temp = gr_list_test[i]
        gt_train_temp = gt_list_train[i]
        gr_train_temp = gr_list_train[i]

        plt.figure(figsize=(20,20))
        plt.imshow(gt_test_temp[0,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(os.path.join(figure_directory, "gt_test_{}.png".format(i+1)))

        plt.figure(figsize=(20,20))
        plt.imshow(gr_test_temp[0,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(os.path.join(figure_directory, "gr_test_{}.png".format(i+1)))

        plt.figure(figsize=(20,20))
        plt.imshow(gt_train_temp[0,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(os.path.join(figure_directory, "gt_train_{}.png".format(i+1)))

        plt.figure(figsize=(20,20))
        plt.imshow(gr_train_temp[0,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(os.path.join(figure_directory, "gr_train_{}.png".format(i+1)))


    