# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 19:52:31 2022

@author: hlinl
"""


import argparse
import os
import glob
import copy
import imageio

import numpy as np
import scipy
import scipy.io
import torch
import torch.nn as nn
import torch.utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL

from torchvision import transforms

from PARAM import *
from dataprocessing import *
from model import *

from files import *
from acoustic_processing import *
from image_processing import *
from training import *
from dataset import *
from test import *


# DEBUG apram. 
DEBUG = False


# JOB PARAMS. 
FRAME_PROCESS_DATA_GENERATION_TOKEN = False
FRAME_CODER_MODEL_TYPE = 'VAE' # Options: 'VAE', 'AE'
VAE_TRAINING_TOKEN = False
# IMAGE_FEATURE_TYPE = 'Hu'

ACOUSTIC_DATA_PROCESSING_TOKEN = True
VISUAL_DATA_PROCESSING_TOKEN = False
ACOUSTIC_VISUAL_MODEL_TYPE = 'conv_2d'
ACOUSTIC_VISUAL_TRAINING_TOKEN = False


if __name__ == "__main__":
    # Set parser.
    parser = argparse.ArgumentParser(description="__init__")

    # Dataset & Data generation. 
    parser.add_argument("--data_dir_path", default=BASIC.DATA_DIRECTORY, type=str, 
                        help="The directory of all raw data, including audio and image. ")
    
    # Melting visual image & frame variables & param. 
    parser.add_argument("--img_data_subdir", default=BASIC.IMAGE_DATA_SUBDIR, type=str, 
                        help="The folder name of raw image data, including subfolders of different layers. ")
    parser.add_argument("--img_processed_data_subdir", default=BASIC.IMAGE_PROCESSED_DATA_SUBDIR, type=str, 
                        help="The folder name of processed image data, including subfolders of different layers. ")
    parser.add_argument("--img_extension", default=IMG.IMAGE_EXTENSION, type=str, 
                        help="The extension (file format) of raw image files. ")
    parser.add_argument("--image_size", default=IMG.IMAGE_SIZE, type=list, 
                        help="The intended size ([h, w]) of the processed image. ")
    parser.add_argument("--img_straighten_keyword", default=IMG.IMG_STRAIGHTEN_KEYWORD, type=str, 
                        help="The keyword of image straighten mode for the class `Frame`. ")
    parser.add_argument("--frame_align_mode", default=IMG.FRAME_ALIGN_MODE, type=str, 
                        help="The keyword indicating the moving axis of the melt pool image frame. ")
    parser.add_argument("--frame_realign_axis_vect", default=IMG.FRAME_REALIGN_AXIS_VECT, type=str, 
                        help="The keyword indicating the targeted axis of the melt pool image frame. ")
    parser.add_argument("--is_binary_frame", default=IMG.IS_BINARY, type=bool, 
                        help="Indicate whether the processed images require binarization. ")
    
    # Acoustic data variables & param. 
    parser.add_argument("--acoustic_data_subdir", default=BASIC.AUDIO_DATA_SUBDIR, type=str,
                        help="The folder name of raw acoustic data, including subfolders of different layers. ")
    parser.add_argument("--photodiode_data_subdir", default=BASIC.PHOTODIODE_DATA_SUBDIR, type=str,
                        help="The folder name of raw acoustic data, including subfolders of different layers. ")
    parser.add_argument("--acoustic_processed_data_subdir", default=BASIC.ACOUSTIC_PROCESSED_DATA_SUBFOLDER, type=str, 
                        help="The folder name of processed acoustic data, including subfolders of different layers. ")
    parser.add_argument("--acoustic_extension", default=ACOUSTIC.AUDIO_FILE_EXTENSION, type=str, 
                        help="The extension (file format) of raw acoustic files. ")
    parser.add_argument("--acoustic_clips_folder_path", default=BASIC.ACOUSTIC_CLIPS_FOLDER_PATH, type=str, 
                        help="The directory of saved clips. ")
    parser.add_argument("--acoustic_specs_folder_path", default=BASIC.ACOUSTIC_SPECS_FOLDER_PATH, type=str, 
                        help="The directory of saved (short) wavelet spectrums. ")

    # Frame coder model param. (VAE & AE) 
    parser.add_argument("--vae_input_img_dir", default=ML_VAE.INPUT_IMAGE_DATA_DIR, type=str, 
                        help="The directory of all input images for VAE. ")
    parser.add_argument("--vae_output_img_dir", default=ML_VAE.OUTPUT_IMAGE_DATA_DIR, type=str, 
                        help="The directory of all output images for VAE. ")
    parser.add_argument("--vae_batch_size", default=ML_VAE.BATCH_SIZE, type=int, 
                        help="The batch size of dataset for VAE. ")
    parser.add_argument("--vae_lr", default=ML_VAE.LEARNING_RATE, type=float, 
                        help="The (initial) learning rate of training for VAE. ")
    parser.add_argument("--vae_num_epochs", default=ML_VAE.NUM_EPOCHS, type=int, 
                        help="The epoch numbers of training for VAE. ")
    parser.add_argument("--vae_loss_beta", default=ML_VAE.LOSS_BETA, type=float, 
                        help="Hyperparameter for controlling weights of the KL-divergence loss function term. ")
    
    # Visual data processing. 
    parser.add_argument("--visual_data_subdir", default=BASIC.VISUAL_DATA_SUBDIR, type=str,
                        help="The directory of visual data. ")

    # Acoustic to visual feature - cnn2d - param. 
    parser.add_argument("--cnn2d_input_data_dir", default=ML_2DCONV.INPUT_WAVELET_SHORT_DIR, type=str, 
                        help="The directory of all input acoustic spectrum (short wavelet) data for conv_2d. ")
    parser.add_argument("--cnn2d_output_data_dir", default=ML_2DCONV.OUTPUT_VISUAL_DIR, type=str, 
                        help="The directory of all output visual features for conv_2d. ")
    parser.add_argument("--cnn2d_batch_size", default=ML_2DCONV.BATCH_SIZE, type=int, 
                        help="The batch size of dataset for conv_2d. ")
    parser.add_argument("--cnn2d_lr", default=ML_2DCONV.LEARNING_RATE, type=float, 
                        help="The (initial) learning rate of training for conv_2d. ")
    parser.add_argument("--cnn2d_num_epochs", default=ML_2DCONV.NUM_EPOCHS, type=int, 
                        help="The epoch numbers of training for conv_2d. ")
    parser.add_argument("--cnn2d_loss_plot_savepath", default=ML_2DCONV.TRAIN_VALID_LOSS_SAVEPATH, type=str, 
                        help="The save path for conv_2d loss plot. ")
                        

    args = parser.parse_args()

    ####################################################################################################################

    # Data processing - Image and acoustic. 
    if FRAME_PROCESS_DATA_GENERATION_TOKEN:
        # Image data processing. 
        img_data_subdir_path = os.path.join(args.data_dir_path, args.img_data_subdir) # Directory of raw melt pool images. 
        img_processed_data_subdir_path = os.path.join(args.data_dir_path, args.img_processed_data_subdir) # Directory of processed melt pool images. 
        clr_dir(img_processed_data_subdir_path)

        img_subfolder_list = os.listdir(img_data_subdir_path) # List of subfolders of different layers.

        for img_subfolder in img_subfolder_list:
            img_processed_subfolder_path = os.path.join(img_processed_data_subdir_path, img_subfolder)
            if not os.path.isdir(img_processed_subfolder_path): os.mkdir(img_processed_subfolder_path) # Create folder for processed images. 

            img_subfolder_path = os.path.join(img_data_subdir_path, img_subfolder)
            img_filepath_perSubFolder_list = glob.glob(os.path.join(img_subfolder_path, 
                                                                    "*.{}".format(args.img_extension))) # List of image file paths of each layer's subfolder. 

            for img_ind, img_filepath in enumerate(img_filepath_perSubFolder_list): # Image data processing part can be separated from the main workflow. 
                frame_temp = Frame(img_filepath)
                meltpool_straightened_image_temp = frame_temp.straighten(args.img_straighten_keyword, 
                                                                         args.frame_align_mode,
                                                                         args.frame_realign_axis_vect) # Get straightened meltpool image.

                img_processed_temp = copy.deepcopy(meltpool_straightened_image_temp)
                if not args.is_binary_frame: img_processed_temp = PIL.Image.fromarray(np.uint8(img_processed_temp*255)) # Not binarizing the image, keeping the original intensity values. 
                else: img_processed_temp = PIL.Image.fromarray(np.uint8(frame_temp.binarize(img_processed_temp)*255)) # Binarize image and convert it to `uint8` data type. 
                img_processed_temp = transforms.Resize(args.image_size)(img_processed_temp)

                img_processed_file_path_temp = os.path.join(img_processed_subfolder_path, 
                                                            "{}_{}.{}".format(img_subfolder, img_ind, args.img_extension))
                img_processed_temp.save(img_processed_file_path_temp)

    ####################################################################################################################

    # Autoencoder training. 
    if VAE_TRAINING_TOKEN:
        if FRAME_CODER_MODEL_TYPE == 'VAE':
            vae_model = Model_VAE(input_data_dir=args.vae_input_img_dir, output_data_dir=args.vae_output_img_dir, 
                                batch_size=args.vae_batch_size, learning_rate=args.vae_lr, 
                                num_epochs=args.vae_num_epochs, loss_beta=args.vae_loss_beta)
            
            # Save dataset repo dicts. 
            np.save("train_set_ind_array.npy", vae_model.train_set_ind_array)
            np.save("valid_set_ind_array.npy", vae_model.valid_set_ind_array)
            np.save("test_set_ind_array.npy", vae_model.test_set_ind_array)

            scipy.io.savemat("input_data_repo_dict.mat", vae_model.dataset.input_data_repo_dict)
            scipy.io.savemat("output_data_repo_dict.mat", vae_model.dataset.output_data_repo_dict)

            vae_model.train() # Train the model. 
            vae_model.loss_plot() # Plot Train & Valid Loss. 

            # In-situ evaluation. Test on outside dataset should be implemented separately. 
            # Validation & data saving process takes roughly 10 mins. 
            (loss_list_test, groundtruths_list_test, generations_list_test,
            mu_list_test, logvar_list_test, latent_list_test) = vae_model.evaluate(vae_model.vae_net, vae_model.test_loader)
            (loss_list_train, groundtruths_list_train, generations_list_train,
            mu_list_train, logvar_list_train, latent_list_train) = vae_model.evaluate(vae_model.vae_net, vae_model.train_loader) 

            np.save("loss_list_test.npy", loss_list_test)
            np.save("groundtruths_list_test.npy", groundtruths_list_test)
            np.save("generations_list_test.npy", generations_list_test)
            np.save("mu_list_test.npy", mu_list_test)
            np.save("logvar_list_test.npy", logvar_list_test)
            np.save("latent_list_test.npy", latent_list_test)

            np.save("loss_list_train.npy", loss_list_train)
            np.save("groundtruths_list_train.npy", groundtruths_list_train)
            np.save("generations_list_train.npy", generations_list_train)
            np.save("mu_list_train.npy", mu_list_train)
            np.save("logvar_list_train.npy", logvar_list_train)
            np.save("latent_list_train.npy", latent_list_train)
        
        elif FRAME_CODER_MODEL_TYPE == 'AE':
            ae_model = Model_AE(input_data_dir=args.vae_input_img_dir, output_data_dir=args.vae_output_img_dir, 
                                batch_size=args.vae_batch_size, learning_rate=args.vae_lr, num_epochs=args.vae_num_epochs)
            
            # Save dataset repo dicts. 
            np.save("train_set_ind_array.npy", ae_model.train_set_ind_array)
            np.save("valid_set_ind_array.npy", ae_model.valid_set_ind_array)
            np.save("test_set_ind_array.npy", ae_model.test_set_ind_array)

            scipy.io.savemat("input_data_repo_dict.mat", ae_model.dataset.input_data_repo_dict)
            scipy.io.savemat("output_data_repo_dict.mat", ae_model.dataset.output_data_repo_dict)

            ae_model.train() # Train the model. 
            ae_model.loss_plot() # Plot Train & Valid Loss. 

            # In-situ evaluation. Test on outside dataset should be implemented separately. 
            # Validation & data saving process takes roughly 10 mins. 
            (loss_list_test, groundtruths_list_test, 
             generations_list_test, latent_list_test) = ae_model.evaluate(ae_model.autoencoder, ae_model.test_loader)
            (loss_list_train, groundtruths_list_train, 
             generations_list_train, latent_list_train) = ae_model.evaluate(ae_model.autoencoder, ae_model.train_loader) 

            np.save("loss_list_test.npy", loss_list_test)
            np.save("groundtruths_list_test.npy", groundtruths_list_test)
            np.save("generations_list_test.npy", generations_list_test)
            np.save("latent_list_test.npy", latent_list_test)

            np.save("loss_list_train.npy", loss_list_train)
            np.save("groundtruths_list_train.npy", groundtruths_list_train)
            np.save("generations_list_train.npy", generations_list_train)
            np.save("latent_list_train.npy", latent_list_train)

    ####################################################################################################################

    # Acoustic data processing. 
    if ACOUSTIC_DATA_PROCESSING_TOKEN:
        audio_file_list = os.listdir(args.acoustic_data_subdir)
        for audio_file in audio_file_list:
            audio_file_name_temp = os.path.splitext(audio_file)[0]
            audio_file_path_temp = os.path.join(args.acoustic_data_subdir, audio_file)
            photodiode_file_path_temp = os.path.join(args.photodiode_data_subdir, audio_file)

            sync_obj_temp = Synchronizer(acoustic_data_file_path=audio_file_path_temp,
                                         photodiode_data_file_path=photodiode_file_path_temp)
            audio_sample = sync_obj_temp.acoustic_synced_data(audio_sensor_No=ACOUSTIC.AUDIO_SENSOR_NO)

            clips_obj = Clips(data=audio_sample, data_label=audio_file_name_temp)
            clips_obj.save_data_offline(is_clips=ACOUSTIC.IS_SAVE_CLIPS, is_spectrums=ACOUSTIC.IS_SAVE_SPECTRUMS)

    ####################################################################################################################

    # Visual data processing. 
    if VISUAL_DATA_PROCESSING_TOKEN:
        collect_visual_data(img_dir=args.img_data_subdir, visual_dir=args.visual_data_subdir, 
                            featurization_mode=IMG.VISUAL_DATA_FEATURIZATION_MODE)
        visual_mean_vect, visual_std_vect = visual_data_standard(visual_dir=args.visual_data_subdir)

        np.save("visual_mean_vect.npy", visual_mean_vect)
        np.save("visual_std_vect.npy", visual_std_vect)

    ####################################################################################################################

    # Mainstream learning. 
    if ACOUSTIC_VISUAL_TRAINING_TOKEN:
        if ACOUSTIC_VISUAL_MODEL_TYPE == 'conv_2d':
            cnn2d_model = Model_Conv_2d(input_data_dir=args.cnn2d_input_data_dir, output_data_dir=args.cnn2d_output_data_dir, 
                                        batch_size=args.cnn2d_batch_size, learning_rate=args.cnn2d_lr, 
                                        num_epochs=args.cnn2d_num_epochs)

            cnn2d_model.train() # Train the model. 
            cnn2d_model.loss_plot(save_path=args.cnn2d_loss_plot_savepath) # Plot Train & Valid Loss. 

            # In-situ evaluation. Test on outside dataset should be implemented separately. 
            # Validation & data saving process takes roughly 10 mins. 
            loss_list_test, \
            groundtruths_list_test, \
            generations_list_test = cnn2d_model.evaluate(cnn2d_model.cnn2d_net, cnn2d_model.test_loader)

            loss_list_train, \
            groundtruths_list_train, \
            generations_list_train = cnn2d_model.evaluate(cnn2d_model.cnn2d_net, cnn2d_model.train_loader)

            loss_list_unseen, \
            groundtruths_list_unseen, \
            generations_list_unseen = cnn2d_model.evaluate(cnn2d_model.cnn2d_net, cnn2d_model.unseen_layers_loader) 

            np.save("loss_list_test.npy", loss_list_test)
            np.save("groundtruths_list_test.npy", groundtruths_list_test)
            np.save("generations_list_test.npy", generations_list_test)

            np.save("loss_list_train.npy", loss_list_train)
            np.save("groundtruths_list_train.npy", groundtruths_list_train)
            np.save("generations_list_train.npy", generations_list_train)

            np.save("loss_list_unseen.npy", loss_list_unseen)
            np.save("groundtruths_list_unseen.npy", groundtruths_list_unseen)
            np.save("generations_list_unseen.npy", generations_list_unseen)
        
        else: pass

