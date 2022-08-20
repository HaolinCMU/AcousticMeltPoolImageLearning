# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 12:44:43 2022

@author: hlinl
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix


def val_to_label(data, thrsld_1=200, thrsld_2=700):
    """
    """
    
    data = np.where(data<=thrsld_1, 0, data)
    data = np.where((data>thrsld_1)&(data<=thrsld_2), 1, data)
    data = np.where(data>thrsld_2, 2, data)
    
    return data


#######################################################

generations_list_test = np.load('C:/Users/hlinl/Desktop/acoustic_image_learning/result/conv_2d/3_08192022_img_256_bs_128_lr_1e-4/generations_list_test.npy')
groundtruths_list_test = np.load('C:/Users/hlinl/Desktop/acoustic_image_learning/result/conv_2d/3_08192022_img_256_bs_128_lr_1e-4/groundtruths_list_test.npy')

generations_label_test = val_to_label(generations_list_test)
groundtruths_label_test = val_to_label(groundtruths_list_test)

r2score_test = r2_score(groundtruths_list_test, generations_list_test)
cfm_test = confusion_matrix(generations_label_test, groundtruths_label_test)

plt.figure()
plt.plot(np.arange(2500), np.arange(2500), color='r')
plt.scatter(generations_list_test, groundtruths_list_test)
plt.title("Test dataset R2 plot")

#######################################################

generations_list_unseen = np.load('C:/Users/hlinl/Desktop/acoustic_image_learning/result/conv_2d/3_08192022_img_256_bs_128_lr_1e-4/generations_list_unseen.npy')
groundtruths_list_unseen = np.load('C:/Users/hlinl/Desktop/acoustic_image_learning/result/conv_2d/3_08192022_img_256_bs_128_lr_1e-4/groundtruths_list_unseen.npy')

generations_label_unseen = val_to_label(generations_list_unseen)
groundtruths_label_unseen = val_to_label(groundtruths_list_unseen)

r2score_unseen = r2_score(groundtruths_list_unseen, generations_list_unseen)
cfm_unseen = confusion_matrix(generations_label_unseen, groundtruths_label_unseen)

plt.figure()
plt.plot(np.arange(2500), np.arange(2500), color='r')
plt.scatter(generations_list_unseen, groundtruths_list_unseen)
plt.title("Unseen full-layer dataset R2 plot")

plt.figure()
plt.plot(groundtruths_list_unseen, color='blue')
plt.plot(generations_list_unseen, color='orange')
plt.title("Unseen full-layer dataset consecutive plot")

#######################################################

generations_list_train = np.load('C:/Users/hlinl/Desktop/acoustic_image_learning/result/conv_2d/3_08192022_img_256_bs_128_lr_1e-4/generations_list_train.npy')
groundtruths_list_train = np.load('C:/Users/hlinl/Desktop/acoustic_image_learning/result/conv_2d/3_08192022_img_256_bs_128_lr_1e-4/groundtruths_list_train.npy')

generations_label_train = val_to_label(generations_list_train)
groundtruths_label_train = val_to_label(groundtruths_list_train)

r2score_train = r2_score(groundtruths_list_train, generations_list_train)
cfm_train = confusion_matrix(generations_label_train, groundtruths_label_train)

plt.figure()
plt.plot(np.arange(2500), np.arange(2500), color='r')
plt.scatter(generations_list_train, groundtruths_list_train)
plt.title("Train dataset R2 plot")