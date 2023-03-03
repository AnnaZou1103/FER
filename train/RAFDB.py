from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data

class RAFDB(data.Dataset):
    """
    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', data_path='../preprocess/raf_db_refined.h5', transform=None):
        self.transform = transform
        self.split = split
        self.data = h5py.File(data_path, 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['training_pixel']
            self.train_labels = self.data['training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((-1, 48, 48, 3))

        elif self.split == 'Test':
            self.test_data = self.data['test_pixel']
            self.test_labels = self.data['test_label']
            self.test_data = np.asarray(self.test_data)
            self.test_data = self.test_data.reshape((-1, 48, 48, 3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Test':
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = img[:, :, np.newaxis]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Test':
            return len(self.test_data)
