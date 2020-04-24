import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import utils
import pdb, cv2


class Dataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        data_df = pd.read_csv(data_dir,
                              names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
        self.X = data_df[['center', 'left', 'right']]
        self.y = data_df['steering']

        self.transforms = transforms


    def __getitem__(self, ind):
        """
        Returns tuple with random one of 3 images taken at a time and adjusted steering angle
        """
        sample = utils.choose_image(self.X.loc[ind, :], self.y.loc[ind])

        if self.transforms is not None:
            sample = self.transforms.augument(sample)
        return sample

    def __len__(self):
        """
        Returns total number of samples in dataset
        """
        return self.y.shape[0]