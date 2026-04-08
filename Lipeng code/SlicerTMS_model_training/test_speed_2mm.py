#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:20:04 2019

@author: gx020
"""

import os
import time
import datetime
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.parallel
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import ntpath
import scipy.io
#from sklearn.model_selection import train_test_split

from model import Modified3DUNet
#from baseline_model import Baseline3DUNet
from Nifti_Load_Tensor import RandomSamples, nifti_numpy_loader
from Load_Data import LoadData as LD
from Loss_Function import LossFun as LF
import Train_Data_Loader as TD

#from medicaltorch import transforms as mt_transforms
#from medicaltorch import datasets as mt_datasets

from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils

from Optim_Method import radam as RD
from collections import OrderedDict
from Loss_Function import CalDice as Dice

## root setting
root_path = '/NAS/Synology/MultiCoilData/BrainCSF/Iso/2mm/Code/testing_R01'
train_path = os.path.join(root_path, 'training_data')
test_path = os.path.join(root_path, 'Coil5') # /media/NAS/NAS/tms_data/testing_data
model_save_path = os.path.join('/NAS/Synology/MultiCoilData/BrainCSF/Iso/2mm/Code/GP_data/code/Network_train_toy/models')
print('model_saved_path:', model_save_path)
  

## loada data
batch_size = 40


## build network
# Loading the model
in_channels = 4
out_channels = 3
base_n_filter = 16
initial_lr = 0.002
net = Modified3DUNet(in_channels, out_channels, base_n_filter) 
net = net.float()
# select gpu
#device = torch.device("cpu") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)

# put the model on GPUs by model.to(device)
net = net.to(device)  
optimizer = RD.RAdam(net.parameters(),lr=initial_lr, betas=(0.9, 0.999), eps=1e-08)
# set learning_rate strategry: Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
criterion = nn.MSELoss().cuda()   

## testing
# load model
model_path = os.path.join(model_save_path, 'Iso_threecoils_noinitial/MultiCoil_MSE_epoch_900.pth.tar')
checkpoint = torch.load(model_path)
# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    name = k # name = k[7:] remove `module.`
#    print(name)
    new_state_dict[name] = v
# load params
net.load_state_dict(new_state_dict)        
#    net.load_state_dict(checkpoint['model_state_dict'])
net = net.cuda()
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
next_epoch = checkpoint['epoch'] + 1 # the next epoch
# call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference
net.eval()


# testing
dataset_test = LD.RandomSamples(test_path, loader=LD.nifti_numpy_loader, 
                                 loader_dataset=LD.make_dataset_subject)
testloader = DataLoader(dataset=dataset_test,
                         batch_size=1, shuffle=False) 
print("[INFO]: Total number of testing samples", len(testloader) * 1)
print('#'*20)
#total_loss = 0.0
#test_loss_path = os.path.join(test_path, 'infor_test_loss_20subj_res_cond.txt')
#test_dice_path = os.path.join(test_path, 'infor_test_dice_20subj_res_cond.txt')
#file_loss_test = open(test_loss_path, 'a')
#file_dice_test = open(test_dice_path, 'a') 
time_started = time.time()

# write starting time
pred_time = os.path.join(root_path, 'pred_time.txt') 
file_pred_time_ = open(pred_time, 'a')

for i, data in enumerate(testloader):
    test_input_path_name = dataset_test.samples[i]
    file_name = ntpath.basename(test_input_path_name)
    under_score = file_name.rfind('_')
    subj_name = file_name[:15]    
    y_inputs, y_target = data[0].to(device), data[1].to(device)   
#    print('#'*10, 'Sample: ', i, '#'*10)
#    print(y_inputs.size())
#    print(y_target.size())
    # predict
    start_time = datetime.datetime.now()
    for j in range(1000):
        pd_out = net(y_inputs.float())
    # finish predict collect time
    finish_time = datetime.datetime.now()
    pred_time = (finish_time-start_time).total_seconds()
    file_pred_time_.write(str(pred_time) + '\n') 

    print('Training complete in {:.0f}m {:.0f}s'.format(
        pred_time // 60, pred_time % 60))
  
    # save the predict data
    move_cpu = pd_out.cpu()
    pd_out_save = move_cpu.detach().numpy()
    # save_pred_path = os.path.join(test_path, 'simu_cond_pd_40_epoch39')
    # save_subj_name = os.path.join(save_pred_path, ('%s_E_pd.mat' % (subj_name)))
    # save it as mat
    # scipy.io.savemat(save_subj_name, dict(pred=pd_out_save))
    
# write end time


