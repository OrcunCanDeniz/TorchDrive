from torch.utils.data import Dataset
import pandas as pd
import utils

class Dataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        data_df = pd.read_csv(data_dir,
                              names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
        self.X = data_df[['center', 'left', 'right', 'speed']]
        self.y = data_df[['steering', 'throttle', 'reverse']]

        self.transforms = transforms


    def __getitem__(self, ind):
        """
        Returns tuple with random one of 3 images taken at a time and adjusted steering angle
        """
        sample = utils.choose_image(self.X.loc[ind], self.y.loc[ind])
        if self.transforms is not None:
            sample = self.transforms.augument(sample)
        return sample

    def __len__(self):
        """
        Returns total number of samples in dataset
        """
        return self.y.shape[0]