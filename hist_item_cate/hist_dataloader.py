#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import os
import sys
import torch
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.distributions as distributions
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
from skimage import io, transform
from skimage.color import rgb2gray
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


class tiny_Dataset(Dataset):
    """Aida-17k dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels, comma .
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csv = pd.read_csv(csv_file, header=None, dtype=str)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # ori
        _ipath = self.csv.iloc[idx, 0]
        img_name = os.path.join(self.root_dir,
                                _ipath)
#         print('idx', idx)
#         print(_ipath)
        image = io.imread(img_name)
        
        _lb = self.csv.iloc[idx, 1]
        label = np.zeros(8)
        label[int(_lb)] = 1.
    
        cls_label = np.array(label)
        sample = image, cls_label
        if self.transform:
            sample = self.transform(sample)

        return sample
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
#         image, label, diqa = sample
        image, label = sample
        h, w = image.shape[:-1]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        image = transform.resize(image, (new_h, new_w))
        return image, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
#         image, label, diqa = sample
        image, label = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
#         image = np.expand_dims(image, axis=2)
        image = image.transpose((2, 0, 1))
        
        return torch.from_numpy(image).type(torch.FloatTensor), torch.from_numpy(label).type(torch.FloatTensor)


# In[ ]:




