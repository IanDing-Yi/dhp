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


from hist_dataloader import tiny_Dataset, Rescale, ToTensor
from pretrain_model import get_pretrain_model


# In[3]:


torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:


def train(clf, optimizer, trainloader, criterion, disp):
    count = 0
    policy_losses = []
    value_losses = []
    episode_reward = []
    if(disp):
        print(device)
    for i, data in enumerate(trainloader, 0):
        count += 1
        if device is None:
            inputs = data[0].type(torch.FloatTensor)
            labels = data[1].type(torch.FloatTensor)
        else:
            inputs = data[0].type(torch.FloatTensor).to(device)
            labels = data[1].type(torch.FloatTensor).to(device)
        
        value_pred = clf(inputs)
        value_loss = criterion(value_pred.float(), labels).sum()
        
#         if(disp):
#             print(value_loss)
        
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()
        
        value_losses.append(float(value_loss.item()))

    return sum(value_losses)/len(value_losses)


# evaluation
def comp_test(stage, clf, testloader, criterion, disp):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    preds = np.empty(0)
    lbs = np.empty(0)
    loss = []
    if(disp):
        print(device)
    with torch.no_grad():
        for data in testloader:
            if device is None:
                inputs = data[0]
                labels = data[1]
            else:
                inputs = data[0].to(device)
                labels = data[1].to(device)

            outputs = clf(inputs)
            val_loss = criterion(outputs.float(), labels).sum()
            loss.append(val_loss)
#             predicted = torch.round(torch.sigmoid(outputs))
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            pred_npy = predicted.detach().cpu().numpy()
            total += labels.size(0)
            labels = torch.argmax(torch.softmax(labels, dim=-1), dim=-1)
            lb_npy = labels.detach().cpu().numpy()
            correct += (pred_npy == lb_npy).sum().item()
            preds = np.hstack((preds, pred_npy.squeeze()))
            lbs = np.hstack((lbs, lb_npy.squeeze()))

    conmx = confusion_matrix(lbs, preds)
    if(disp):
        print(stage+' accuracy: %.6f %%' % (100 * correct / total))
#     tn, fp, fn, tp = conmx.ravel()
#     if (tp + fp) == 0:
#         prec = 0
#     else:
#         prec = tp / (tp + fp)
#     if (tp + fn) == 0:
#         recl = 0
#     else:
#         recl = tp / (tp + fn)
#     if (prec+recl) == 0:
#         f1 = 0
#     else:
#         f1 = (2*prec*recl) / (prec+recl)
#     if(disp):
#         print('Precision:', prec)
#         print('Recall:', recl)
#         print('F1:', f1)
    return (correct / total), conmx, sum(loss)/len(loss)

def run_train(model_name, train_csv, val_csv, root_folder, save_path, disp):
    start_time = time.time()

    test_csv = val_csv

    train_dataset = tiny_Dataset(csv_file=train_csv,
                                 root_dir=root_folder,
                                 transform=transforms.Compose([
                                     Rescale((224,224)),
                                     ToTensor()
                                 ]))
    test_dataset = tiny_Dataset(csv_file=test_csv,
                                root_dir=root_folder,
                                transform=transforms.Compose([
                                    Rescale((224,224)),
                                    ToTensor()
                                ]))

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, num_workers=0)

    clf = get_pretrain_model(model_name)
    clf.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_clf = optim.Adam(clf.parameters(), lr=0.0001)

    max_test_perf = 10000
    min_delta = 0
    patience = 5
    counter = 0

    MAX_EPISODES = 1000
    PRINT_EVERY = 1

    records = {'train': [],'valid': []}
    for episode in range(1, MAX_EPISODES+1):  # loop over the dataset multiple times
        if(disp):
            print('episode:', episode)
        critic_loss = train(clf, optimizer_clf, trainloader, criterion, disp)
        if(disp):
            print('Train')
        tr_cur_acc, tr_conmx, tr_loss = comp_test('Train', clf, trainloader, criterion, disp)
        records['train'].append([tr_cur_acc, tr_conmx, tr_loss])
        if(disp):
            print('train loss: ', critic_loss)
            print('train loss: ', tr_loss)
            print('Validation')
        cur_acc, conmx, val_loss = comp_test('Validation', clf, testloader, criterion, disp)
        records['valid'].append([cur_acc, conmx, val_loss])
        
        if(disp):
            print('validation loss: ', val_loss)

        if max_test_perf - val_loss > min_delta:
            if(disp):
                print('refresh patience')
            max_test_perf = val_loss
            counter = 0
            # save model
            cur_high = [cur_acc, conmx]
            torch.save(clf.state_dict(), save_path)
    #                 print('after  val_loss', val_loss, 'best_loss', best_loss)
        elif max_test_perf - val_loss < min_delta:
#             if (episode > 50):
            if(disp):
                print('patience counter +1')
            counter += 1
            if counter >= patience:
                break

    # print('\t'.join([str(it) for it in [cur_high[3], cur_high[0], cur_high[1], cur_high[2]]]))


    if(disp):
        print('Finished Training')
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    if(disp):
        print(time_elapsed)

    return records

def run_test(model_name, test_csv, root_folder, model_path, disp):
    # run test
    start_time = time.time()

    pth = model_path
    
    test_dataset = tiny_Dataset(csv_file=test_csv,
                                root_dir=root_folder,
                                transform=transforms.Compose([
                                    Rescale((224,224)),
                                    ToTensor()
                                ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, num_workers=0)
    
    clf = get_pretrain_model(model_name)
    clf.load_state_dict(torch.load(pth))
    clf.to(device)
    criterion = nn.BCEWithLogitsLoss()
    cur_acc, conmx, val_loss = comp_test('Test', clf, testloader, criterion, disp)

    if(disp):
        print('Finished Testing')
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    if(disp):
        print(time_elapsed)
    
    return cur_acc, conmx, val_loss


# In[5]:


def run(var_save_name, model_name, model_save_path,
        train_csv, valid_csv, test_csv, base_path,
        run_count = 1, disp = False):
    
    exps_rslts = []
    for iter_count in range(run_count):

        train_records = run_train(model_name,
                                  train_csv,
                                  valid_csv,
                                  base_path,
                                  model_save_path,
                                  disp
                                  )
        print(var_save_name, 'train')
        cur_acc, conmx, val_loss = run_test(model_name,
                                            test_csv,
                                            base_path,
                                            model_save_path,
                                            disp
                                            )
        print(var_save_name, 'test')
        exps_rslts.append([cur_acc, conmx, val_loss, train_records])
        print(cur_acc)
        display(conmx)

        with open('result_data_'+var_save_name+'_'+str(iter_count)+'.pkl', 'wb') as fp:
            pickle.dump(exps_rslts, fp)
            print('exps rslts saved successfully to file: ', iter_count)
        
#         print(var_save_name, iter_count)
#         print(exps_rslts)

