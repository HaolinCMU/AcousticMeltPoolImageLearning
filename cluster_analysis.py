# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 04:06:08 2022

@author: hlinl
"""


import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold


def get_pairwiseDist_2D(matrix, sample_axis):
    """
    matrix: 2D array. 
    sample_axis: 0 or 1. 
    """
    
    # Make the layout of matrix: sampleNum * featureNum. 
    if sample_axis == 0: array = matrix
    else: array = matrix.T
    
    sample_num = array.shape[0]
    P = array @ array.T # sample_num * sample_num. 
    
    K = np.tile(np.diag(P).reshape(1,-1), (sample_num, 1))
    
    return np.sqrt(K + K.T - 2*P) # sample_num * sample_num. 


def tSNE_plot_2D(data, figure_name="tSNE_plot.png"):
    """
    """
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(data)

    color_plot_list = []
    for i in range(data.shape[0]):
        if i == 0: color_plot_list.append('red')
        else: color_plot_list.append('blue')
    
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    plt.scatter(Y[:,0], Y[:,1], c=color_plot_list, cmap=plt.cm.Spectral, linewidths=10.0)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(figure_name)


def tSNE_plot_2D_cvalue(data, label_array, figure_name="tSNE_plot.png"):
    """
    """
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(data)
    # label_array = (np.array(label_array) - np.mean(label_array)) / np.std(label_array)
    
    plt.figure(figsize=(20,20))
    plt.rcParams.update({"font.size": 35})
    plt.tick_params(labelsize=35)
    plt.scatter(Y[:,0], Y[:,1], c=label_array, cmap=plt.cm.coolwarm, linewidths=10.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(orientation="vertical")
    
    plt.savefig(figure_name)


if __name__ == "__main__":
    """
    """
    result_directory = "result/22"
    save_directory = os.path.join(result_directory, "similarities")

    IOI = 20

    mu_list_test = np.load(os.path.join(result_directory, "mu_list_test.npy"))
    # latent_list_test = np.load(os.path.join(result_directory, "latent_list_test.npy"))
    generation_list_test = np.load(os.path.join(result_directory, "generations_list_test.npy"))

    mu_list_test = mu_list_test.reshape(mu_list_test.shape[0], -1)[:51,:]
    mu_l2dist_matrix = get_pairwiseDist_2D(mu_list_test, sample_axis=0)

    index_list = list(np.arange(mu_l2dist_matrix.shape[1]))
    dist_list_OI = [mu_l2dist_matrix[IOI,i] for i in range(mu_l2dist_matrix.shape[1])]

    zipped = list(zip(dist_list_OI, index_list))
    zipped.sort(key=lambda x: x[0], reverse=False)
    dist_list_OI, index_list = list(zip(*zipped))
    index_list, dist_list_OI = copy.deepcopy(list(index_list)), copy.deepcopy(np.array(list(dist_list_OI)).reshape(-1))

    ############################################################################

    SOI = [0,1,2,3] + [int(mu_l2dist_matrix.shape[1]/2)-1, int(mu_l2dist_matrix.shape[1]/2), int(mu_l2dist_matrix.shape[1]/2)+1] + \
          [mu_l2dist_matrix.shape[1]-3, mu_l2dist_matrix.shape[1]-2, mu_l2dist_matrix.shape[1]-1]

    gr_test_OI = generation_list_test[IOI]

    plt.figure(figsize=(20,20))
    plt.imshow(gr_test_OI[0,:,:], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(os.path.join(save_directory, "gr_IOI_{}_dist_0.png".format(IOI)))

    for i in SOI:
        gr_test_temp = generation_list_test[index_list[i]]

        plt.figure(figsize=(20,20))
        plt.imshow(gr_test_temp[0,:,:], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.savefig(os.path.join(save_directory, "gr_IOI_{}_SOI_{}_dist_{:.2f}.png".format(IOI, i+1, dist_list_OI[i])))

    tSNE_plot_2D_cvalue(mu_list_test[[index_list[i] for i in SOI]], dist_list_OI[SOI], 
                        figure_name=os.path.join(save_directory, "tSNE_plot_IOI_{}.png".format(IOI)))
        

    
