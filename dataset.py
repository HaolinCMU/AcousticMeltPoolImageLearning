# -*- coding: utf-8 -*-
"""
Created on Sun May 22 04:33:47 2022

@author: hlinl
"""


import copy
import glob
import os
import math

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mig
import PIL

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

from PARAM import *


class ACVD_SubDataset(Dataset):
    """
    """
    
    def __init__(self, dataset_dict=None, dtype=torch.float32, 
                 input_img_transform=Compose([Resize([ML_2DCONV.IMG_SIZE,ML_2DCONV.IMG_SIZE]),ToTensor()])):
        """
        """
        
        self.dataset = dataset_dict
        self.dtype = dtype
        self.input_img_transform = input_img_transform
        
        self._dataset_size = len(self.dataset) if self.dataset is not None else 0
    
    
    def __len__(self):
        """
        """
        
        return self._dataset_size

    
    def __getitem__(self, ind):
        """
        """
        
        if self.dataset is None: raise ValueError("Dataset not defined. ")
        else:
            if len(self.dataset) == 0: raise ValueError("No data is found in the data dictionary. ")

            input_img = PIL.Image.fromarray(np.uint8(mig.imread(self.dataset[ind][0])*255)) # Read spectrum from a path.
            
            visual_data_full = np.load(self.dataset[ind][1]).reshape(-1)
            label_array = np.array([visual_data_full[i] for i in ML_2DCONV.VISUAL_FEATURE_LIST]).reshape(-1)
            output_label = torch.from_numpy(label_array).to(self.dtype) # Read label vector from a path.
            
            if self.input_img_transform:
                input_img = copy.deepcopy(self.input_img_transform(input_img).to(self.dtype)) # Transformed tensor of prescribed data type. [c, h, w]. 
            
            sample = {'input': input_img, 'output': output_label}
            return sample
        

class AcousticSpectrumVisualDataset(Dataset):
    """
    """

    def __init__(self, spectrum_data_dir, visual_data_dir, dtype=torch.float32, 
                 train_ratio=0.8, valid_ratio=0.05, test_ratio=0.15, test_layer_folder_namelist=None, 
                 spectrum_extension=ACOUSTIC.SPECTRUM_FIG_EXTENSION, visual_data_extension=IMG.VISUAL_DATA_EXTENSION, 
                 input_image_transform=Compose([Resize([ML_2DCONV.IMG_SIZE,ML_2DCONV.IMG_SIZE]),ToTensor()])):
        """
        `spectrum_data_dir` and `visual_data_dir` must have the same inner data structure. 
        """
        
        self.spectrum_dir = spectrum_data_dir
        self.visual_data_dir = visual_data_dir
        
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.test_layer_folder_namelist = test_layer_folder_namelist
        
        self.dtype = dtype
        self.spectrum_extension = spectrum_extension
        self.visual_data_extension = visual_data_extension
        self.input_image_transform = input_image_transform
        
        self._dataset_size = 0
        self._train_data_dict = defaultdict()
        self._valid_data_dict = defaultdict()
        self._test_data_dict = defaultdict()
        self._testlayers_data_dict = defaultdict()

        self._train_set_obj = None
        self._valid_set_obj = None
        self._test_set_obj = None
        self._testlayers_set_obj = None

        self._train_set_size = None
        self._valid_set_size = None
        self._test_set_size = None
        self._unseen_layers_set_size = None
        
        self._set_data_repo()
    
    
    def __len__(self):
        """
        """
        
        return self._dataset_size

    
    def __getitem__(self):
        """
        """
        
        pass

    
    @property
    def train_set(self):
        """
        """
        
        if len(self._train_data_dict) == 0: 
            if self._dataset_size != 0 and int(np.floor(self.train_ratio*self._dataset_size)) <= 0:
                raise ValueError("Training dataset not established. ")
            else: return None
        else: 
            if self._train_set_obj is None:
                self._train_set_obj = ACVD_SubDataset(dataset_dict=self._train_data_dict, dtype=self.dtype, 
                                                      input_img_transform=self.input_image_transform)
            return self._train_set_obj
    
    
    @property
    def valid_set(self):
        """
        """
        
        if len(self._valid_data_dict) == 0: 
            if self._dataset_size != 0 and int(np.floor(self.valid_ratio*self._dataset_size)) <= 0:
                raise ValueError("Validation dataset not established. ")
            else: return None
        else: 
            if self._valid_set_obj is None:
                self._valid_set_obj = ACVD_SubDataset(dataset_dict=self._valid_data_dict, dtype=self.dtype, 
                                                      input_img_transform=self.input_image_transform)
            return self._valid_set_obj
    
    
    @property
    def test_set(self):
        """
        """
        
        if len(self._test_data_dict) == 0: 
            if self._dataset_size != 0 and int(np.floor(self.test_ratio*self._dataset_size)) <= 0:
                raise ValueError("Test dataset not established. ")
            else: return None
        else: 
            if self._test_set_obj is None:
                self._test_set_obj = ACVD_SubDataset(dataset_dict=self._test_data_dict, dtype=self.dtype, 
                                                     input_img_transform=self.input_image_transform)
            return self._test_set_obj
    
    
    @property
    def unseen_layers_set(self):
        """
        """
        
        if len(self._testlayers_data_dict) == 0: 
            if self.test_layer_folder_namelist is None: 
                raise ValueError("Unseen layers dataset not established. ")
            else: return None
        else: 
            if self._testlayers_set_obj is None: 
                self._testlayers_set_obj = ACVD_SubDataset(dataset_dict=self._testlayers_data_dict, dtype=self.dtype, 
                                                           input_img_transform=self.input_image_transform)
            return self._testlayers_set_obj
    

    @property
    def train_set_size(self):
        """
        """

        if self._train_set_size is not None: return self._train_set_size
        else: raise ValueError("Training dataset not established. ")
    

    @property
    def valid_set_size(self):
        """
        """

        if self._valid_set_size is not None: return self._valid_set_size
        else: raise ValueError("Validation dataset not established. ")

    
    @property
    def test_set_size(self):
        """
        """

        if self._test_set_size is not None: return self._test_set_size
        else: raise ValueError("Testing dataset not established. ")

    
    @property
    def unseen_layers_set_size(self):
        """
        """

        if self._unseen_layers_set_size is not None: return self._unseen_layers_set_size
        else: return 0

    
    @staticmethod
    def _test_valid_train_indices(dataset_size, test_ratio, valid_ratio, train_ratio, shuffle=False):
        """
        """
        
        clips_indices_total = list(range(dataset_size))
        if shuffle: np.random.shuffle(clips_indices_total)
        split_pt_test = int(np.floor(test_ratio*dataset_size))
        split_pt_valid = int(np.floor((test_ratio+valid_ratio)*dataset_size))
        
        train_set_indices = clips_indices_total[split_pt_valid:]
        valid_set_indices = clips_indices_total[split_pt_test:split_pt_valid]
        test_set_indices = clips_indices_total[:split_pt_test]
        
        return test_set_indices, valid_set_indices, train_set_indices
    
    
    def _single_layer_dataparser(self, layer_folder_name, 
                                 train_samples_accum_num, valid_samples_accum_num, test_samples_accum_num,
                                 train_dict, valid_dict, test_dict, 
                                 train_ratio, valid_ratio, test_ratio):
        """
        """

        P, V = ACOUSTIC.extract_process_param_fromAcousticFilename(layer_folder_name)
        
        spectrum_dir = os.path.join(self.spectrum_dir, layer_folder_name)
        visual_dir = os.path.join(self.visual_data_dir, layer_folder_name)
        
        spectrum_pathlist = glob.glob(os.path.join(spectrum_dir, "*.{}".format(self.spectrum_extension)))
        visual_pathlist = glob.glob(os.path.join(visual_dir, "*.{}".format(self.visual_data_extension)))
        
        clips_num = min(len(spectrum_pathlist), len(visual_pathlist)) # Must be the smaller number of spectrums and visuals!
        test_ind_list, valid_ind_list, train_ind_list = self._test_valid_train_indices(clips_num, test_ratio, valid_ratio, 
                                                                                       train_ratio, shuffle=False)
        
        for test_ind in test_ind_list:
            test_dict[test_samples_accum_num] = [spectrum_pathlist[test_ind], visual_pathlist[test_ind],
                                                 layer_folder_name, str(test_ind).zfill(5), P, V]
            test_samples_accum_num += 1
        
        for valid_ind in valid_ind_list:
            valid_dict[valid_samples_accum_num] = [spectrum_pathlist[valid_ind], visual_pathlist[valid_ind],
                                                   layer_folder_name, str(valid_ind).zfill(5), P, V]
            valid_samples_accum_num += 1
        
        for train_ind in train_ind_list:
            train_dict[train_samples_accum_num] = [spectrum_pathlist[train_ind], visual_pathlist[train_ind],
                                                   layer_folder_name, str(train_ind).zfill(5), P, V]
            train_samples_accum_num += 1
        
        return train_dict, valid_dict, test_dict, train_samples_accum_num, valid_samples_accum_num, test_samples_accum_num
    
    
    def _set_data_repo(self):
        """
        """
        
        layers_folder_name_list = os.listdir(self.spectrum_dir) # Should be the same if using `self.visual_data_dir`. 
        
        if self.test_layer_folder_namelist is not None:
            testlayer_set_samplenum = 0
            
            for testlayer_foldername in self.test_layer_folder_namelist:
                layers_folder_name_list.remove(testlayer_foldername)
                
                self._testlayers_data_dict, _, _, \
                testlayer_set_samplenum, _, _ = self._single_layer_dataparser(testlayer_foldername, testlayer_set_samplenum, 
                                                                              0, 0, self._testlayers_data_dict, 
                                                                              {}, {}, 1., 0., 0.)
                
            self._unseen_layers_set_size = copy.deepcopy(testlayer_set_samplenum)

        else: pass
        
        train_set_samplenum, valid_set_samplenum, test_set_samplenum = 0, 0, 0
        for layer_folder_name in layers_folder_name_list:
            self._train_data_dict, self._valid_data_dict, \
            self._test_data_dict, train_set_samplenum, \
            valid_set_samplenum, test_set_samplenum = self._single_layer_dataparser(layer_folder_name, train_set_samplenum, 
                                                                                    valid_set_samplenum, test_set_samplenum,
                                                                                    self._train_data_dict, self._valid_data_dict, 
                                                                                    self._test_data_dict, self.train_ratio, 
                                                                                    self.valid_ratio, self.test_ratio)
        
        self._train_set_size = copy.deepcopy(train_set_samplenum)
        self._valid_set_size = copy.deepcopy(valid_set_samplenum)
        self._test_set_size = copy.deepcopy(test_set_samplenum)
    
    
    def subdataset(self, mode):
        """
        """
        
        if mode == "train": return self.train_set
        elif mode == "valid": return self.valid_set
        elif mode == "test": return self.test_set
        elif mode == "unseen": return self.unseen_layers_set
        else: raise ValueError("Wrong mode input. ")


class FrameAutoencoderDataset(Dataset):
    """
    """

    def __init__(self, input_data_dir, output_data_dir, img_pattern=IMG.IMAGE_EXTENSION, dtype=torch.float32, 
                 input_image_transform=Compose([Resize(ML_VAE.INPUT_IMAGE_SIZE), ToTensor()]), 
                 output_image_transform=Compose([Resize(ML_VAE.OUTPUT_IMAGE_SIZE), ToTensor()])):
        """
        Expected data directory structure: folder (data_dir) -> subfolders (layers) -> images (frames). 
        """

        self.input_data_dir = input_data_dir # The total directory of all input images for the autoencoder. 
        self.output_data_dir = output_data_dir # The total directory of all output images for the autoencoder. 
        self.img_pattern = img_pattern
        self.image_dtype = dtype
        self.input_image_transform = input_image_transform
        self.output_image_transform = output_image_transform
        
        # Data repos (path lists) & Data label dictionaries. 
        self._dataset_size = 0 
        self._input_image_label_dict = {} # {`ind`->int: [`subfolder`->str, `image_No`->int, `image_path`-> str]}. 
        self._output_image_label_dict = {} # {`ind`->int: [`subfolder`->str, `image_No`->int, `image_path`-> str]}. 

        self._set_data_repo()
    

    def __len__(self):
        """
        """

        return self._dataset_size


    def __getitem__(self, index):
        """
        """

        index = copy.deepcopy(str(int(index)))

        # input_img = PIL.Image.open(self._input_image_label_dict[index][2])
        # output_img = PIL.Image.open(self._output_image_label_dict[index][2])

        input_img = PIL.Image.fromarray(np.uint8(mig.imread(self._input_image_label_dict[index][2])*255))
        output_img = PIL.Image.fromarray(np.uint8(mig.imread(self._output_image_label_dict[index][2])*255))

        if self.input_image_transform:
            input_img = copy.deepcopy(self.input_image_transform(input_img).to(self.image_dtype)) # Transformed tensor of prescribed data type. [c, h, w]. 
        if self.output_image_transform:
            output_img = copy.deepcopy(self.output_image_transform(output_img).to(self.image_dtype)) # Transformed tensor of prescribed data type. [c, h, w]. 

        # # ---------- Reshape tensors to make them compatible with NN ---------- 
        # input_img_c, input_img_h, input_img_w = input_img.size()
        # output_img_c, output_img_h, output_img_w = output_img.size()

        # input_img = input_img.view(-1, input_img_c, input_img_h, input_img_w)
        # output_img = output_img.view(-1, output_img_c, output_img_h, output_img_w)

        sample = {'input': input_img, 'output': output_img}

        return sample
    

    @property
    def input_data_repo_dict(self):
        """
        """

        return self._input_image_label_dict
    

    @property
    def output_data_repo_dict(self):
        """
        """

        return self._output_image_label_dict
    

    def _set_data_repo(self):
        """
        """

        # ---------- Set input data repo ---------- 
        input_image_subfolder_list = os.listdir(self.input_data_dir)
        input_image_subdir_list = glob.glob(os.path.join(self.input_data_dir, "*"))
        input_image_accum_num = 0 # Used for establishing the image label dict. 

        for ind, input_image_subdir in enumerate(input_image_subdir_list):
            input_image_subfolder_name_temp = input_image_subfolder_list[ind]
            input_image_path_list_temp = copy.deepcopy(glob.glob(os.path.join(input_image_subdir, 
                                                                              "*.{}".format(self.img_pattern))))

            # Establish image label dict. 
            for i, path in enumerate(input_image_path_list_temp):
                self._input_image_label_dict[str(int(input_image_accum_num+i))] = [input_image_subfolder_name_temp, i, path]

            input_image_accum_num += len(input_image_path_list_temp)

        # ---------- Set output data repo ---------- 
        output_image_subfolder_list = os.listdir(self.output_data_dir)
        output_image_subdir_list = glob.glob(os.path.join(self.output_data_dir, "*"))
        output_image_accum_num = 0 # Used for establishing the image label dict. 

        for ind, output_image_subdir in enumerate(output_image_subdir_list):
            output_image_subfolder_name_temp = output_image_subfolder_list[ind]
            output_image_path_list_temp = copy.deepcopy(glob.glob(os.path.join(output_image_subdir, 
                                                                               "*.{}".format(self.img_pattern))))

            # Establish image label dict. 
            for i, path in enumerate(output_image_path_list_temp):
                self._output_image_label_dict[str(int(output_image_accum_num+i))] = [output_image_subfolder_name_temp, i, path]

            output_image_accum_num += len(output_image_path_list_temp)
        
        self._dataset_size = min(input_image_accum_num, output_image_accum_num)


    def extract(self, ind_array):
        """
        Return info of input and output data that corresponds to the given `ind_array` with the exact order. 
        """

        ind_array = copy.deepcopy(ind_array.astype(str).reshape(-1))
        
        # Input. 
        input_image_subfolder_list = [self._input_image_label_dict[ind][0] for ind in ind_array]
        input_image_ind_list = [self._input_image_label_dict[ind][1] for ind in ind_array]
        input_image_path_list = [self._input_image_label_dict[ind][2] for ind in ind_array]

        # Output. 
        output_image_subfolder_list = [self._output_image_label_dict[ind][0] for ind in ind_array]
        output_image_ind_list = [self._output_image_label_dict[ind][1] for ind in ind_array]
        output_image_path_list = [self._output_image_label_dict[ind][2] for ind in ind_array]

        return (input_image_subfolder_list, input_image_ind_list, input_image_path_list,
                output_image_subfolder_list, output_image_ind_list, output_image_path_list)

