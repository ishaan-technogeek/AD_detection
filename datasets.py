from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import utils_custom as utils
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
import multiprocessing
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision

from tabulate import tabulate


# Binary brain mask used to cut out the skull.
mask = utils.load_nifti('C:/AD_Detection/binary_brain_mask.nii')


# ------------------------- ADNI data tables -----------------------------------

ADNI_DIR = 'data/ADNI'

# Filepaths for 3 Tesla scans.
table_3T = r'C:\AD_Detection_v2\ADNI1_Screening_1.5T_8_25_2024_Complete.csv'
image_dir_3T = r'D:\ADNI data'
corrupt_images_3T = ['037_S_0501/Baseline', '037_S_0501/Month12', '037_S_0501/Month24', '051_S_1123/Baseline', '051_S_1123/Month12', '051_S_1123/Month24', '116_S_0649/Month12', '116_S_0649/Month24', '116_S_1232/Baseline', '027_S_1387/Baseline', '027_S_1387/Month12', '027_S_1387/Month24', '116_S_0382/Baseline', '027_S_0404/Baseline', '027_S_0404/Month24', '027_S_1385/Month12', '023_S_0376/Month12', '023_S_0030/Baseline', '023_S_0030/Month24', '023_S_1247/Baseline', '023_S_1247/Month12', '027_S_1082/Month24', '018_S_0450/Baseline', '005_S_0572/Baseline', '005_S_0572/Month12', '005_S_0572/Month24']

# Filepaths for 1.5 Tesla scans.
table_15T = r'C:\AD_Detection_v2\ADNI1_Screening_1.5T_8_25_2024_Complete.csv'
image_dir_15T = r'D:\ADNI data'
corrupt_images_15T = []


def load_data_table(table, image_dir, corrupt_images=None):
    """Read data table, find corresponding images, filter out corrupt, missing and MCI images, and return the samples as a pandas dataframe."""
    
    # Read table into dataframe.
    print('Loading dataframe for', table)
    df = pd.read_csv(table)
    print('Found', len(df), 'images in table')
    
    # Use the 'img_path' column for file paths.
    df['filepath'] = df['filepath'].apply(lambda path: image_dir + path.replace('/content/drive/MyDrive', '')).replace('/','\\')

    # Filter out corrupt images (i.e. images where the preprocessing failed).
    len_before = len(df)
    if corrupt_images is not None:
        df = df[df.apply(lambda row: '{}/{}'.format(row['Subject'], row['Visit']) not in corrupt_images, axis=1)]
    print('Filtered out', len_before - len(df), 'of', len_before, 'images because of failed preprocessing')
    
    # Filter out images where files are missing.
    len_before = len(df)
    df = df[df['filepath'].map(os.path.exists)]
    print('Filtered out', len_before - len(df), 'of', len_before, 'images because of missing files')
    
    # Filter out images with MCI. (Not Needed as we do not have any MCI data with us)
    # len_before = len(df)
    # df = df[df['Group'] != 'MCI']
    # print('Filtered out', len_before - len(df), 'of', len_before, 'images that were MCI')
    
    print('Final dataframe contains', len(df), 'images from', len(df['Subject'].unique()), 'patients')
    print()
    
    # df = df.sample(120, replace=True, random_state=42)

    return df


def load_data_table_3T():
    """Load the data table for all 3 Tesla images."""
    return load_data_table(table_3T, image_dir_3T, corrupt_images_3T)
    
def load_data_table_15T():
    """Load the data table for all 1.5 Tesla images."""
    return load_data_table(table_15T, image_dir_15T, corrupt_images_15T)
    
def load_data_table_both():
    """Load the data tables for all 1.5 Tesla and 3 Tesla images and combine them."""
    df_15T = load_data_table(table_15T, image_dir_15T, corrupt_images_15T)
    df_3T = load_data_table(table_3T, image_dir_3T, corrupt_images_3T)
    df = pd.concat([df_15T, df_3T])
    return df


# ------------------------ PyTorch datasets and loaders ----------------------

def pad_or_crop_image(image, target_shape):
    # Calculate padding or cropping sizes
    pad_x = max(target_shape[0] - image.shape[0], 0)
    pad_y = max(target_shape[1] - image.shape[1], 0)
    pad_z = max(target_shape[2] - image.shape[2], 0)

    # Apply padding
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        pad_width = ((0, pad_x), (0, pad_y), (0, pad_z))
        image = np.pad(image, pad_width, mode='constant', constant_values=0)

    # Apply cropping
    crop_x = max(image.shape[0] - target_shape[0], 0)
    crop_y = max(image.shape[1] - target_shape[1], 0)
    crop_z = max(image.shape[2] - target_shape[2], 0)

    if crop_x > 0 or crop_y > 0 or crop_z > 0:
        start_x = crop_x // 2
        start_y = crop_y // 2
        start_z = crop_z // 2
        end_x = start_x + target_shape[0]
        end_y = start_y + target_shape[1]
        end_z = start_z + target_shape[2]
        image = image[start_x:end_x, start_y:end_y, start_z:end_z]

    return image

class ADNIDataset(Dataset):
    """
    PyTorch dataset that consists of MRI images and labels.
    
    Args:
        filenames (iterable of strings): The filenames fo the MRI images.
        labels (iterable): The labels for the images.
        mask (array): If not None (default), images are masked by multiplying with this array.
        transform: Any transformations to apply to the images.
    """

    def __init__(self, filenames, labels, mask=None, transform=None):
        self.filenames = filenames
        self.labels = torch.LongTensor(labels)
        self.mask = mask
        self.transform = transform

        # Required by torchsample.
        self.num_inputs = 1
        self.num_targets = 1

        # Default values. Should be set via fit_normalization.
        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Return the image as a numpy array and the label."""
        label = self.labels[idx]

        struct_arr = utils.load_nifti(self.filenames[idx], mask=self.mask)
        # TDOO: Try normalizing each image to mean 0 and std 1 here.
        #struct_arr = (struct_arr - struct_arr.mean()) / (struct_arr.std() + 1e-10)
        struct_arr = (struct_arr - self.mean) / (self.std + 1e-10)  # prevent 0 division by adding small factor
        struct_arr = struct_arr[None]  # add (empty) channel dimension
        struct_arr = torch.FloatTensor(struct_arr)

        if self.transform is not None:
            struct_arr = self.transform(struct_arr)

        return struct_arr, label

    def image_shape(self):
        """The shape of the MRI images."""
        return utils.load_nifti(self.filenames[0]).shape

    def fit_normalization(self, num_sample=None, show_progress=False):
        """
        Calculate the voxel-wise mean and std across the dataset for normalization.
        
        Args:
            num_sample (int or None): If None (default), calculate the values across the complete dataset, 
                                      otherwise sample a number of images.
            show_progress (bool): Show a progress bar during the calculation."
        """
            
        if num_sample is None:
            num_sample = len(self)

        num_sample = min(num_sample, len(self.filenames))  

        image_shape = self.image_shape()
        all_struct_arr = np.zeros((num_sample, image_shape[0], image_shape[1], image_shape[2]))

        sampled_filenames = np.random.choice(self.filenames, num_sample, replace=False)
        if show_progress:
            sampled_filenames = tqdm_notebook(sampled_filenames)

        for i, filename in enumerate(sampled_filenames):
            struct_arr = utils.load_nifti(filename, mask=mask)
            all_struct_arr[i] = struct_arr

        self.mean = all_struct_arr.mean(0)
        self.std = all_struct_arr.std(0)

    def get_raw_image(self, idx):
        """Return the raw image at index idx (i.e. not normalized, no color channel, no transform."""
        return utils.load_nifti(self.filenames[idx], mask=self.mask)
        
def print_df_stats(df, df_train, df_val):
    """Print some statistics about the patients and images in a dataset."""
    headers = ['Images', '-> AD', '-> CN', 'Patients', '-> AD', '-> CN']

    def get_stats(df):
        # Ensure column names and values are correct
        if 'Subject' in df.columns:
            df_ad = df[df['Group'] == 'AD']
            df_cn = df[df['Group'] == 'CN']
            num_images = len(df)
            num_ad_images = len(df_ad)
            num_cn_images = len(df_cn)
            num_patients = len(df['Subject'].unique())
            num_ad_patients = len(df_ad['Subject'].unique())
            num_cn_patients = len(df_cn['Subject'].unique())
        else:
            print("Column 'Subject' not found in dataframe.")
            return [0, 0, 0, 0, 0, 0]

        return [num_images, num_ad_images, num_cn_images, num_patients, num_ad_patients, num_cn_patients]

    # Collect statistics for all, train, and validation sets
    stats = []
    stats.append(['All'] + get_stats(df))
    stats.append(['Train'] + get_stats(df_train))
    stats.append(['Val'] + get_stats(df_val))

    # Print the table with statistics
    print(tabulate(stats, headers=headers, tablefmt='grid'))
    print()


# TODO: Rename *_val to *_test.
def build_datasets(df, patients_train, patients_val, print_stats=True, normalize=False):
    """
    Build PyTorch datasets based on a data table and a patient-wise train-test split.
    
    Args:
        df (pandas dataframe): The data table from ADNI.
        patients_train (iterable of strings): The patients to include in the train set.
        patients_val (iterable of strings): The patients to include in the val set.
        print_stats (boolean): Whether to print some statistics about the datasets.
        normalize (boolean): Whether to caluclate mean and std across the dataset for later normalization.
        
    Returns:
        The train and val dataset.
    """
    # Compile train and val dfs based on patients.
    df_train = df[df.apply(lambda row: row['Subject'] in patients_train, axis=1)]
    df_val = df[df.apply(lambda row: row['Subject'] in patients_val, axis=1)]

    if print_stats:
        print_df_stats(df, df_train, df_val)

    # Extract filenames and labels from dfs.
    train_filenames = np.array(df_train['filepath'])
    val_filenames = np.array(df_val['filepath'])
    train_labels = np.array(df_train['Group'] == 'AD', dtype=int)#[:, None]
    val_labels = np.array(df_val['Group'] == 'AD', dtype=int)#[:, None]

    train_dataset = ADNIDataset(train_filenames, train_labels)
    val_dataset = ADNIDataset(val_filenames, val_labels)

    # TODO: Maybe normalize each scan first, so that they are on a common scale.
    # TODO: Save these values to file together with the model.
    # TODO: Sample over more images.

    if normalize:
        print('Calculating mean and std for normalization:')
        train_dataset.fit_normalization(40, show_progress=True)
        val_dataset.mean, val_dataset.std = train_dataset.mean, train_dataset.std
    else:
        # print('Dataset is not normalized, this could dramatically decrease performance')
        print('Dataset is already normalized')

    return train_dataset, val_dataset


def build_loaders(train_dataset, val_dataset):
    """Build PyTorch data loaders from the datasets."""
    
    # In contrast to Korolev et al. 2017, we do not enforce one sample per class in each batch.
    # TODO: Maybe change batch size to 3 or 4. Check how this affects memory and accuracy.
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=multiprocessing.cpu_count(), pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=multiprocessing.cpu_count(), pin_memory=torch.cuda.is_available())

    return train_loader, val_loader
