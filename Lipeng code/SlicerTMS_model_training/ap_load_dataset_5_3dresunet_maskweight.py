#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:07:38 2019

@author: jeg88
"""

import os
import time
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.parallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
#from sklearn.model_selection import train_test_split

from model import Modified3DUNet
#from baseline_model import Baseline3DUNet
from Nifti_Load_Tensor import RandomSamples, nifti_numpy_loader
from Load_Data import LoadData as LDM
from Loss_Function import LossFun as LF
import Train_Data_Loader as TD

from medicaltorch import transforms as mt_transforms
from medicaltorch import datasets as mt_datasets

from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils

## root setting
root_path = '/rfanfs/pnl-zorro/home/gp88/tms_data'
#train_path = '/rfanfs/pnl-zorro/home/gp88/software/anaconda3/doc/guoping_test/pytorch-3dunet/tms_data'
test_path = '/rfanfs/pnl-zorro/home/gp88/software/anaconda3/doc/guoping_test/proj1_doc/simu_data'
model_fold = 'model_save/3dres-unet_mask_2.pt'
model_save_path = os.path.join('/rfanfs/pnl-zorro/home/gp88/software/anaconda3/doc/guoping_test/proj1_doc', model_fold)
print('model_path', model_save_path)
  

## loada data
batch_size = 4
#train_transform = transforms.Compose([
#        mt_transforms.RandomRotation3D(10)
#        ])
dataset_train = LDM.RandomSamplesMask(root_path, loader=LDM.nifti_numpy_loader,
                               transform=None)
train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True)
# load a subject, check a subject
# Note: here the batch is dict
sample, target, mask = next(iter(train_loader))
print("Sample size: ", sample.size())
print("Target size:", target.size())
print("Mask size:", mask.size())
print("[INFO]: Total number of trainings samples", len(train_loader) * batch_size)


#import pdb
#pdb.set_trace()

## build network
# Loading the model
in_channels = 6
out_channels = 3
base_n_filter = 16
net = Modified3DUNet(in_channels, out_channels, base_n_filter) 
net = net.float()
#print(net)
# training on multi-GPUs
#torch.cuda.empty_cache() 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cuda:1"
#print(device)   
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)
# put the model on GPUs by model.to(device)
net = net.to(device)  

import torch.nn.functional as F
## another wey to define loss
def cal_mask_loss(output, target, mask):
    output = output * mask
    loss = F.mse_loss(output, target, reduction='mean')
    return loss

## training model function
def train_model(trainloader, model, criterion, optimizer, scheduler, num_epochs=25):
    time_started = time.time()  
    num_steps = 0
    # save log
    writer = SummaryWriter(log_dir="log_exp_mask_2")    
    for epoch in tqdm(range(num_epochs)):
        start_time_epoch = time.time()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)        
        running_loss = 0.0    
        train_loss_total = 0.0
        num_steps = 0
        for i, data in enumerate(trainloader, 0): # index = 0
            inputs, labels, masks = data[0].to(device), data[1].to(device), data[2].to(device)
            masks = masks.float() 
            labels = labels.float()
            masks[masks < torch.max(masks)] = 0.8 # set a weigth for the region outside the GM : 0.5, 0.6
            
#            masks.to(device)
            # forward
            outputs = net(inputs.float())
            outputs_weight = outputs * masks
#            labels_weight = labels * masks
            # another way
            loss = cal_mask_loss(outputs_weight, labels, masks)
##            loss = LF.MaskWeighedLoss.cal_loss(criterion, outputs, labels.float(), masks.float())
##            loss = LF.MaskWeighedLoss(criterion(outputs, labels.float()), masks.float())
#            loss = criterion(outputs_weight, labels)
            # zero the params gradients
            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loss_total += loss.item()
            num_steps += 1
            
            if (i+1) % 10 == 0:    # print every 2000 mini-batches
                print('[Epoch: %d/%d, Batch index: %5d] loss: %.3f' %
                      (epoch + 1, num_epochs, i + 1, running_loss))
                running_loss = 0.0            
        
        train_loss_total_avg = train_loss_total / num_steps  
        writer.add_scalars('losses', {'train_loss': train_loss_total_avg}, epoch)
        
        # check time
        end_time_epoch = time.time()
        total_time_epoch = end_time_epoch - start_time_epoch
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch+1, total_time_epoch))
      
        # save the model after 10 epochs: Epoch from 0 to end
        if epoch % 9 == 0 or epoch == (num_epochs-1):
            print('[INFO]:Epoch-', epoch + 1)
            torch.save(net, model_save_path)
            
    print('Finished Training') 
    train_mean_loss = train_loss_total / (len(trainloader)*(num_epochs))
    print('Mean Loss: ', train_mean_loss)    
    print('#' * 20)
    time_elapsed = time.time() - time_started
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('#' * 20)    
    return model, time_elapsed, train_mean_loss
    

# do training  
# define loss func and optim
num_epochs = 100
initial_lr = 0.002
#criterion = nn.MSELoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(),lr=initial_lr, betas=(0.9, 0.999), eps=1e-08)
# set learning_rate strategry: Decay LR by a factor of 0.1 every 100 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
criterion = nn.MSELoss()

## train and save the model path
print('#'*10, 'Training', '#'*10)
net, time_elapsed, train_mean_loss = train_model(train_loader, net, criterion, 
                                optimizer, exp_lr_scheduler, num_epochs=num_epochs)      

## testing
# load model
#net = torch.load(model_save_path)
## call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference
#net.eval()
dataset_test = RandomSamples(test_path, 'testing', 'simu_target', 
                                loader = nifti_numpy_loader)
testloader = DataLoader(dataset=dataset_test,
                         batch_size=1, shuffle=False) 
total_loss = 0.0
for i, data in enumerate(testloader):
    y_inputs, y_target = data[0].to(device), data[1].to(device)   
    print('#'*10, 'Sample: ', i, '#'*10)
    print(y_inputs.size())
    print(y_target.size())
    pd_out = net(y_inputs.float())
    pd_loss = criterion(pd_out, y_target.float())
    print('Loss: ', pd_loss.item())
    total_loss += pd_loss.item()
test_mean_loss =  total_loss / len(testloader)  
print("Mean loss:", total_loss / (i+1))  

## post-precessing: save the related information in training and testing
# write revalant information to a text file
infor_record = {'used_time': time_elapsed, 'train_mean_loss': train_mean_loss,
                'test_mean_loss':test_mean_loss}
f = open('infor_recored.txt', 'w')
f.write('dict = ' + repr(infor_record) + '\n')
f.close()
print('#'*10, 'Done', '#'*10)