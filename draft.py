# -*- coding: utf-8 -*-
"""
Created on Wed May 25 04:12:00 2022

@author: hlinl
"""


import os
import glob
import copy
import re

import numpy as np
import matplotlib as mpl
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


COLORS_MAP_LIST = ['red', 'green', 'blue', 'orange', 'purple', 
                   'pink', 'gray', 'cyan', 'brown', 'olive'] # 10 different colors. 

layer_num_list = [3, 12, 17, 22, 27, 32, 37, 42, 47, 57, 62, 67, 72, 77, 82, 87, 92, 97, 
                  102, 107, 112, 117, 122, 132, 137, 142, 147, 152, 157]
P_list = [280, 280, 300, 180, 200, 280, 280, 220, 280, 280, 165, 370, 150, 200, 250, 300, 350, 
          280, 280, 280, 280, 280, 280, 280, 240, 260, 320, 340, 165]
V_list = [1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 400, 1200, 1585, 640, 850, 1070, 1280, 
          1500, 1200, 1200, 1200, 600, 800, 1000, 1600, 1200, 1200, 1200, 1200, 1200]
H_list = [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.12, 0.1, 
          0.08, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.08]

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
    
    colors_map_list = COLORS_MAP_LIST

    color_plot_list = []
    for i in range(data.shape[0]):
        color_plot_list.append(colors_map_list[label_array[i]%len(colors_map_list)])
    
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    plt.scatter(Y[:,0], Y[:,1], c=color_plot_list, cmap=plt.cm.Spectral, linewidths=1.0)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(figure_name)


def plot_2D(data, label_array, figure_name="plot_2D.png"):
    """
    """
    
    colors_map_list = COLORS_MAP_LIST

    color_plot_list = []
    for i in range(data.shape[0]):
        color_plot_list.append(colors_map_list[label_array[i]%len(colors_map_list)])
    
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    plt.scatter(data[:,0], data[:,1], c=color_plot_list, cmap=plt.cm.Spectral, linewidths=1.0)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(figure_name)


def tSNE_plot_2D_cvalue(data, label_array, figure_name="tSNE_plot.png"):
    """
    """
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(data)
    # norm = mpl.colors.Normalize(vmin=np.min(label_array), vmax=np.max(label_array))
    # label_array = (np.array(label_array) - np.min(label_array))/(np.max(label_array) - np.min(label_array))
    label_array = (np.array(label_array) - np.mean(label_array)) / np.std(label_array)
    
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    plt.scatter(Y[:,0], Y[:,1], c=label_array, cmap=plt.cm.coolwarm, linewidths=1.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(orientation="vertical")
    
    plt.savefig(figure_name)


def plot_2D_cvalue(data, label_array, figure_name="plot_2D.png"):
    """
    """
    
    # label_array = (np.array(label_array) - np.min(label_array))/(np.max(label_array) - np.min(label_array))
    label_array = (np.array(label_array) - np.mean(label_array)) / np.std(label_array)
    
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    plt.scatter(data[:,0], data[:,1], c=label_array, cmap=plt.cm.coolwarm, linewidths=1.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(orientation="vertical")
    
    plt.savefig(figure_name)


if __name__ == "__main__":
    result_directory = "result/20"
    figure_directory = os.path.join(result_directory, "figures")
    cluster_directory = os.path.join(result_directory, "clusters")
    
    is_VAE = True

    if not os.path.isdir(figure_directory): os.mkdir(figure_directory)
    if not os.path.isdir(cluster_directory): os.mkdir(cluster_directory)

    input_data_repo_dict_path = os.path.join(result_directory, "input_data_repo_dict.mat")
    output_data_repo_dict_path = os.path.join(result_directory, "output_data_repo_dict.mat")

    groundtruths_list_test_path = os.path.join(result_directory, "groundtruths_list_test.npy")
    generations_list_test_path = os.path.join(result_directory, "generations_list_test.npy")
    groundtruths_list_train_path = os.path.join(result_directory, "groundtruths_list_train.npy")
    generations_list_train_path = os.path.join(result_directory, "generations_list_train.npy")

    latent_list_test_path = os.path.join(result_directory, "latent_list_test.npy")
    if is_VAE: 
        mu_list_test_path = os.path.join(result_directory, "mu_list_test.npy")
    latent_list_train_path = os.path.join(result_directory, "latent_list_train.npy")
    if is_VAE: 
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
    if is_VAE:
        mu_list_test = np.load(mu_list_test_path)
    latent_list_train = np.load(latent_list_train_path)
    if is_VAE:
        mu_list_train = np.load(mu_list_train_path)
    

    test_set_ind = np.load(test_set_ind_array_path)
    train_set_ind = np.load(train_set_ind_array_path)
    valid_set_ind = np.load(valid_set_ind_array_path)

    input_data_repo_dict = scipy.io.loadmat(input_data_repo_dict_path)
    output_data_repo_dict = scipy.io.loadmat(output_data_repo_dict_path)

    # ===

    n_clusters = 10
    PC_num = 10
    demo_sample_num = 20

    # ===
    
    for i in range(demo_sample_num):
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
    
    
    # Extract P-V for each test datapoint. 
    P_test_list, V_test_list, PV_test_list, PVH_test_list = [], [], [], []
    area_test_list, ar_test_list, spatter_num_test_list, avg_spatter_dist_test_list = [], [], [], []
    for ind in test_set_ind:
        layer_num_string = output_data_repo_dict[str(ind)][0]
        layer_num = int(re.findall(r"\d+", layer_num_string)[0])
        P_temp = P_list[layer_num_list.index(layer_num)]
        V_temp = V_list[layer_num_list.index(layer_num)]
        H_temp = H_list[layer_num_list.index(layer_num)]
        
        frame_temp = Frame(file_path=output_data_repo_dict[str(ind)][2], 
                           intensity_threshold=(0.5,1.0))
        
        P_test_list.append(P_temp)
        V_test_list.append(V_temp)
        PV_test_list.append(P_temp/V_temp)
        PVH_test_list.append(P_temp/(H_temp*V_temp))
        
        area_test_list.append(frame_temp.meltpool_area)
        ar_test_list.append(frame_temp.meltpool_length/frame_temp.meltpool_width
                            if frame_temp.meltpool_width != 0. else 0.)
        spatter_num_test_list.append(frame_temp.spatters_num)
        avg_spatter_dist_test_list.append(np.mean(frame_temp.spatters_distances) 
                                          if frame_temp.spatters_num != 0 else 0.)
    
    
    # pca_encoder_latent_test = pca.PCA(latent_list_test.reshape(latent_list_test.shape[0],-1), 
    #                                   PC_num=PC_num, mode='transpose')
    # latent_test_data = pca_encoder_latent_test.weights
    latent_test_data = latent_list_test.reshape(latent_list_test.shape[0], -1)
    if latent_test_data.shape[1] == 2: 
        plot_2D_cvalue(latent_test_data, P_test_list, 
                        figure_name=os.path.join(figure_directory, "latent_test_P.png"))
        plot_2D_cvalue(latent_test_data, V_test_list, 
                        figure_name=os.path.join(figure_directory, "latent_test_V.png"))
        plot_2D_cvalue(latent_test_data, PV_test_list, 
                        figure_name=os.path.join(figure_directory, "latent_test_PV.png"))
        plot_2D_cvalue(latent_test_data, PVH_test_list, 
                        figure_name=os.path.join(figure_directory, "latent_test_PVH.png"))
        plot_2D_cvalue(latent_test_data, area_test_list, 
                        figure_name=os.path.join(figure_directory, "latent_test_mp_area.png"))
        plot_2D_cvalue(latent_test_data, ar_test_list, 
                        figure_name=os.path.join(figure_directory, "latent_test_mp_ar.png"))
        plot_2D_cvalue(latent_test_data, spatter_num_test_list, 
                        figure_name=os.path.join(figure_directory, "latent_test_sp_num.png"))
        plot_2D_cvalue(latent_test_data, avg_spatter_dist_test_list, 
                        figure_name=os.path.join(figure_directory, "latent_test_sp_avg_dist.png"))
        
    else: 
        tSNE_plot_2D_cvalue(latent_test_data, P_test_list, 
                            figure_name=os.path.join(figure_directory, "latent_test_tsne_P.png"))
        tSNE_plot_2D_cvalue(latent_test_data, V_test_list, 
                            figure_name=os.path.join(figure_directory, "latent_test_tsne_V.png"))
        tSNE_plot_2D_cvalue(latent_test_data, PV_test_list, 
                            figure_name=os.path.join(figure_directory, "latent_test_tsne_PV.png"))
        tSNE_plot_2D_cvalue(latent_test_data, PVH_test_list, 
                            figure_name=os.path.join(figure_directory, "latent_test_tsne_PVH.png"))
        tSNE_plot_2D_cvalue(latent_test_data, area_test_list, 
                            figure_name=os.path.join(figure_directory, "latent_test_mp_area.png"))
        tSNE_plot_2D_cvalue(latent_test_data, ar_test_list, 
                            figure_name=os.path.join(figure_directory, "latent_test_mp_ar.png"))
        tSNE_plot_2D_cvalue(latent_test_data, spatter_num_test_list, 
                            figure_name=os.path.join(figure_directory, "latent_test_sp_num.png"))
        tSNE_plot_2D_cvalue(latent_test_data, avg_spatter_dist_test_list, 
                            figure_name=os.path.join(figure_directory, "latent_test_sp_avg_dist.png"))
    
    if is_VAE:
        # pca_encoder_mu_test = pca.PCA(mu_list_test.reshape(mu_list_test.shape[0],-1), 
        #                               PC_num=PC_num, mode='transpose')
        # mu_test_data = pca_encoder_mu_test.weights
        mu_test_data = mu_list_test.reshape(mu_list_test.shape[0], -1)
        if mu_test_data.shape[1] == 2:
            plot_2D_cvalue(mu_test_data, P_test_list, 
                            figure_name=os.path.join(figure_directory, "mu_test_P.png"))
            plot_2D_cvalue(mu_test_data, V_test_list, 
                            figure_name=os.path.join(figure_directory, "mu_test_V.png"))
            plot_2D_cvalue(mu_test_data, PV_test_list, 
                            figure_name=os.path.join(figure_directory, "mu_test_PV.png"))
            plot_2D_cvalue(mu_test_data, PVH_test_list, 
                            figure_name=os.path.join(figure_directory, "mu_test_PVH.png"))
            plot_2D_cvalue(mu_test_data, area_test_list, 
                            figure_name=os.path.join(figure_directory, "mu_test_mp_area.png"))
            plot_2D_cvalue(mu_test_data, ar_test_list, 
                            figure_name=os.path.join(figure_directory, "mu_test_mp_ar.png"))
            plot_2D_cvalue(mu_test_data, spatter_num_test_list, 
                            figure_name=os.path.join(figure_directory, "mu_test_sp_num.png"))
            plot_2D_cvalue(mu_test_data, avg_spatter_dist_test_list, 
                            figure_name=os.path.join(figure_directory, "mu_test_sp_avg_dist.png"))
        else:
            tSNE_plot_2D_cvalue(mu_test_data, P_test_list, 
                                figure_name=os.path.join(figure_directory, "mu_test_tsne_P.png"))
            tSNE_plot_2D_cvalue(mu_test_data, V_test_list, 
                                figure_name=os.path.join(figure_directory, "mu_test_tsne_V.png"))
            tSNE_plot_2D_cvalue(mu_test_data, PV_test_list, 
                                figure_name=os.path.join(figure_directory, "mu_test_tsne_PV.png"))
            tSNE_plot_2D_cvalue(mu_test_data, PVH_test_list, 
                                figure_name=os.path.join(figure_directory, "mu_test_tsne_PVH.png"))
            tSNE_plot_2D_cvalue(mu_test_data, area_test_list, 
                                figure_name=os.path.join(figure_directory, "mu_test_mp_area.png"))
            tSNE_plot_2D_cvalue(mu_test_data, ar_test_list, 
                                figure_name=os.path.join(figure_directory, "mu_test_mp_ar.png"))
            tSNE_plot_2D_cvalue(mu_test_data, spatter_num_test_list, 
                                figure_name=os.path.join(figure_directory, "mu_test_sp_num.png"))
            tSNE_plot_2D_cvalue(mu_test_data, avg_spatter_dist_test_list, 
                                figure_name=os.path.join(figure_directory, "mu_test_sp_avg_dist.png"))
        
    # # Extract P-V for each training datapoint. 
    # P_train_list, V_train_list, PV_train_list = [], [], []
    # for ind in train_set_ind:
    #     layer_num_string = output_data_repo_dict[str(ind)][0]
    #     layer_num = int(re.findall(r"\d+", layer_num_string)[0])
    #     P_temp = P_list[layer_num_list.index(layer_num)]
    #     V_temp = V_list[layer_num_list.index(layer_num)]
        
    #     P_train_list.append(P_temp)
    #     V_train_list.append(V_temp)
    #     PV_train_list.append(P_temp/V_temp)
    
    # latent_train_data = latent_list_train.reshape(latent_list_train.shape[0], -1)
    # if latent_train_data.shape[1] == 2: 
    #     plot_2D_cvalue(latent_train_data, P_train_list, 
    #                    figure_name=os.path.join(figure_directory, "latent_train_P.png"))
    #     plot_2D_cvalue(latent_train_data, V_train_list, 
    #                    figure_name=os.path.join(figure_directory, "latent_train_V.png"))
    #     plot_2D_cvalue(latent_train_data, PV_train_list, 
    #                    figure_name=os.path.join(figure_directory, "latent_train_PV.png"))
    # else: 
    #     tSNE_plot_2D_cvalue(latent_train_data, P_train_list, 
    #                         figure_name=os.path.join(figure_directory, "latent_train_tsne_P.png"))
    #     tSNE_plot_2D_cvalue(latent_train_data, V_train_list, 
    #                         figure_name=os.path.join(figure_directory, "latent_train_tsne_V.png"))
    #     tSNE_plot_2D_cvalue(latent_train_data, PV_train_list, 
    #                         figure_name=os.path.join(figure_directory, "latent_train_tsne_PV.png"))
    
    # if is_VAE:
    #     mu_train_data = mu_list_train.reshape(mu_list_train.shape[0], -1)
    #     if mu_train_data.shape[1] == 2:
    #         plot_2D_cvalue(mu_train_data, P_train_list, 
    #                        figure_name=os.path.join(figure_directory, "mu_train_P.png"))
    #         plot_2D_cvalue(mu_train_data, V_train_list, 
    #                        figure_name=os.path.join(figure_directory, "mu_train_V.png"))
    #         plot_2D_cvalue(mu_train_data, PV_train_list, 
    #                        figure_name=os.path.join(figure_directory, "mu_train_PV.png"))
    #     else:
    #         tSNE_plot_2D_cvalue(mu_train_data, P_train_list, 
    #                             figure_name=os.path.join(figure_directory, "mu_train_tsne_P.png"))
    #         tSNE_plot_2D_cvalue(mu_train_data, V_train_list, 
    #                             figure_name=os.path.join(figure_directory, "mu_train_tsne_V.png"))
    #         tSNE_plot_2D_cvalue(mu_train_data, PV_train_list, 
    #                             figure_name=os.path.join(figure_directory, "mu_train_tsne_PV.png"))
    
    
    
    # Test set clustering & t-SNE. 
    latent_test_data = latent_list_test.reshape(latent_list_test.shape[0], -1)
    kmeans_latent_test = kmeans_clustering(latent_test_data, n_clusters)
    kmeans_label_array_latent_test, kmeans_center_array_latent_test = kmeans_latent_test.labels_, kmeans_latent_test.cluster_centers_
    if latent_test_data.shape[1] == 2: 
        plot_2D(latent_test_data, kmeans_label_array_latent_test, 
                figure_name=os.path.join(figure_directory, "latent_test_c={}.png".format(n_clusters)))
    else: 
        tSNE_plot_2D(latent_test_data, kmeans_label_array_latent_test, 
                    figure_name=os.path.join(figure_directory, "latent_test_tsne_c={}.png".format(n_clusters)))
    
    if is_VAE:
        mu_test_data = mu_list_test.reshape(mu_list_test.shape[0], -1)
        kmeans_mu_test = kmeans_clustering(mu_test_data, n_clusters)
        kmeans_label_array_mu_test, kmeans_center_array_mu_test = kmeans_mu_test.labels_, kmeans_mu_test.cluster_centers_
        if mu_test_data.shape[1] == 2:
            plot_2D(mu_test_data, kmeans_label_array_mu_test, 
                    figure_name=os.path.join(figure_directory, "mu_test_c={}.png".format(n_clusters)))
        else:
            tSNE_plot_2D(mu_test_data, kmeans_label_array_mu_test, 
                          figure_name=os.path.join(figure_directory, "mu_test_tsne_c={}.png".format(n_clusters)))

    # Copy files only for test dataset. 
    clr_dir(cluster_directory)
    for i in range(n_clusters): 
        subfolder_path_temp = os.path.join(cluster_directory, "{}_{}".format(i, COLORS_MAP_LIST[i%len(COLORS_MAP_LIST)]))
        if not os.path.isdir(subfolder_path_temp): os.mkdir(subfolder_path_temp)

        if is_VAE:
            cluster_temp = test_set_ind[np.where(kmeans_label_array_mu_test==i)] # Move files according to clustering results of `mu`.
        else:
            cluster_temp = test_set_ind[np.where(kmeans_label_array_latent_test==i)] # Move files according to clustering results of `latent`.

        for ind in cluster_temp:
            shutil.copy(output_data_repo_dict[str(ind)][2], subfolder_path_temp+'/')


    # Training set clustering & t-SNE. 
    latent_train_data = latent_list_train.reshape(latent_list_train.shape[0], -1)
    kmeans_latent_train = kmeans_clustering(latent_train_data, n_clusters)
    kmeans_label_array_latent_train, kmeans_center_array_latent_train = kmeans_latent_train.labels_, kmeans_latent_train.cluster_centers_
    if latent_train_data.shape[1] == 2: 
        plot_2D(latent_train_data, kmeans_label_array_latent_train, 
                figure_name=os.path.join(figure_directory, "latent_train_c={}.png".format(n_clusters)))
    else:
        tSNE_plot_2D(latent_train_data, kmeans_label_array_latent_train, 
                    figure_name=os.path.join(figure_directory, "latent_train_tsne_c={}.png".format(n_clusters)))
    
    if is_VAE:
        mu_train_data = mu_list_train.reshape(mu_list_train.shape[0], -1)
        kmeans_mu_train = kmeans_clustering(mu_train_data, n_clusters)
        kmeans_label_array_mu_train, kmeans_center_array_mu_train = kmeans_mu_train.labels_, kmeans_mu_train.cluster_centers_
        if mu_train_data.shape[1] == 2:
            plot_2D(mu_train_data, kmeans_label_array_mu_train, 
                    figure_name=os.path.join(figure_directory, "mu_train_c={}.png".format(n_clusters)))
        else:
            tSNE_plot_2D(mu_train_data, kmeans_label_array_mu_train, 
                        figure_name=os.path.join(figure_directory, "mu_train_tsne_c={}.png".format(n_clusters)))
    
    
    # PCA-clustering & visualization on test dataset. 
    pca_encoder_latent_test = pca.PCA(latent_list_test.reshape(latent_list_test.shape[0],-1), 
                                      PC_num=PC_num, mode='transpose')
    latent_test_data = pca_encoder_latent_test.weights
    kmeans_latent_test = kmeans_clustering(latent_test_data, n_clusters)
    kmeans_label_array_latent_test, kmeans_center_array_latent_test = kmeans_latent_test.labels_, kmeans_latent_test.cluster_centers_
    if latent_test_data.shape[1] == 2: 
        plot_2D(latent_test_data, kmeans_label_array_latent_test, 
                figure_name=os.path.join(figure_directory, "latent_test_pca_c={}.png".format(n_clusters)))
    else: 
        tSNE_plot_2D(latent_test_data, kmeans_label_array_latent_test, 
                     figure_name=os.path.join(figure_directory, "latent_test_pca_tsne_c={}.png".format(n_clusters)))
    
    if is_VAE:
        pca_encoder_mu_test = pca.PCA(mu_list_test.reshape(mu_list_test.shape[0],-1), 
                                      PC_num=PC_num, mode='transpose')
        mu_test_data = pca_encoder_mu_test.weights
        kmeans_mu_test = kmeans_clustering(mu_test_data, n_clusters)
        kmeans_label_array_mu_test, kmeans_center_array_mu_test = kmeans_mu_test.labels_, kmeans_mu_test.cluster_centers_
        if mu_test_data.shape[1] == 2:
            plot_2D(mu_test_data, kmeans_label_array_mu_test, 
                    figure_name=os.path.join(figure_directory, "mu_test_pca_c={}.png".format(n_clusters)))
        else:
            tSNE_plot_2D(mu_test_data, kmeans_label_array_mu_test, 
                         figure_name=os.path.join(figure_directory, "mu_test_pca_tsne_c={}.png".format(n_clusters)))
            
    
    # # Copy files only for test dataset. 
    # clr_dir(cluster_directory)
    # for i in range(n_clusters): 
    #     subfolder_path_temp = os.path.join(cluster_directory, "{}_{}".format(i, COLORS_MAP_LIST[i%len(COLORS_MAP_LIST)]))
    #     if not os.path.isdir(subfolder_path_temp): os.mkdir(subfolder_path_temp)

    #     if is_VAE:
    #         cluster_temp = test_set_ind[np.where(kmeans_label_array_mu_test==i)] # Move files according to clustering results of `mu`.
    #     else:
    #         cluster_temp = test_set_ind[np.where(kmeans_label_array_latent_test==i)] # Move files according to clustering results of `latent`.

    #     for ind in cluster_temp:
    #         shutil.copy(output_data_repo_dict[str(ind)][2], subfolder_path_temp+'/')
    


    