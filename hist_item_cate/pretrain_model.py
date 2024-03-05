#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import torchvision
import torchvision.models as models


# In[ ]:


def get_resnet50(output_shape):

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.resnet50_ft = models.resnet50(pretrained=True)

            self.relu1 = nn.ReLU()
            self.new_fc = nn.Linear(in_features=1000, out_features=output_shape, bias=True)
            

        def forward(self, x):
            x = self.resnet50_ft(x)
            x = self.relu1(x)
            x = self.new_fc(x)

            return x
    
    clf = Classifier()
    
    return clf


# In[ ]:


def get_vgg16(output_shape):

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.vgg16_ft = models.vgg16(pretrained=True)

            self.relu1 = nn.ReLU()
            self.new_fc = nn.Linear(in_features=1000, out_features=output_shape, bias=True)
            

        def forward(self, x):
            x = self.vgg16_ft(x)
            x = self.relu1(x)
            x = self.new_fc(x)

            return x
    
    clf = Classifier()
    
    return clf


# In[ ]:


def get_resnext50(output_shape):

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.resnext50_32x4d_ft = models.resnext50_32x4d(pretrained=True)

            self.relu1 = nn.ReLU()
            self.new_fc = nn.Linear(in_features=1000, out_features=output_shape, bias=True)
            

        def forward(self, x):
            x = self.resnext50_32x4d_ft(x)
            x = self.relu1(x)
            x = self.new_fc(x)

            return x
    
    clf = Classifier()
    
    return clf


# In[ ]:


def get_alexnet(output_shape):

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.alexnet_ft = models.alexnet(pretrained=True)

            self.relu1 = nn.ReLU()
            self.new_fc = nn.Linear(in_features=1000, out_features=output_shape, bias=True)
            

        def forward(self, x):
            x = self.alexnet_ft(x)
            x = self.relu1(x)
            x = self.new_fc(x)

            return x
    
    clf = Classifier()
    
    return clf


# In[ ]:


def get_efficientnet_b0(output_shape):

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.efficientnet_b0_ft = models.efficientnet_b0(pretrained=True)

            self.relu1 = nn.ReLU()
            self.new_fc = nn.Linear(in_features=1000, out_features=output_shape, bias=True)
            

        def forward(self, x):
            x = self.efficientnet_b0_ft(x)
            x = self.relu1(x)
            x = self.new_fc(x)

            return x
    
    clf = Classifier()
    
    return clf


# In[ ]:


def get_pretrain_model(name):
    if name == 'resnet50':
        return get_resnet50(8)
    elif name == 'vgg16':
        return get_vgg16(8)
    elif name == 'resnext50':
        return get_resnext50(8)
    elif name == 'alexnet':
        return get_alexnet(8)
    elif name == 'efficientnet_b0':
        return get_efficientnet_b0(8)
    else:
        return None


# In[ ]:


# print(get_pretrain_model('efficientnet_b0'))


# In[ ]:




