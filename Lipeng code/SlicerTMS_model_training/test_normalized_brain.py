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
import ntpath
import scipy.io
#from sklearn.model_selection import train_test_split

from model import Modified3DUNet
#from baseline_model import Baseline3DUNet
from Nifti_Load_Tensor import RandomSamples, nifti_numpy_loader
from Load_Data import LoadData as LD
from Loss_Function import LossFun as LF
import Train_Data_Loader as TD

from medicaltorch import transforms as mt_transforms
from medicaltorch import datasets as mt_datasets

from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils

from Optim_Method import radam as RD
from collections import OrderedDict
from Loss_Function import CalDice as Dice

## root setting
root_path = '/media/NAS/NAS/tms_data'
train_path = os.path.join(root_path, 'training_data')
test_path = '/rfanfs/pnl-zorro/home/lipeng/NAS/QNAP/MultiCoilData/AnisoCondData/GP_data/validation/normalized_brain/' # os.path.join(root_path, 'testing_data') # /media/NAS/NAS/tms_data/testing_data
model_save_path = os.path.join('/rfanfs/pnl-zorro/home/lipeng/NAS/QNAP/MultiCoilData/AnisoCondData/GP_data/code/Network_train_toy/models/normalized_brain/')
print('model_saved_path:', model_save_path)
  

## loada data
batch_size = 2

## build network
# Loading the model
in_channels = 6
out_channels = 3
base_n_filter = 16
initial_lr = 0.002
net = Modified3DUNet(in_channels, out_channels, base_n_filter) 
net = net.float()

# select gpu


# select gpu
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
device = torch.device('cuda:1')



# put the model on GPUs by model.to(device)
net = net.to(device)  
optimizer = RD.RAdam(net.parameters(),lr=initial_lr, betas=(0.9, 0.999), eps=1e-08)
# set learning_rate strategry: Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
criterion = nn.MSELoss().cuda(device)

## testing
# load model
model_path = os.path.join(model_save_path, 'model_normalized_epoch_30.pth.tar')
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
net = net.cuda(device)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
next_epoch = checkpoint['epoch'] + 1 # the next epoch
# call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference
net.eval()



dataset_test = LD.RandomSamples(test_path, loader=LD.nifti_numpy_loader, 
                                 loader_dataset=LD.make_dataset_subject)
testloader = DataLoader(dataset=dataset_test,
                         batch_size=1, shuffle=False) 
total_loss = 0.0
test_loss_path = os.path.join(test_path, 'infor_test_loss_20subj_res_cond.txt')
test_dice_path = os.path.join(test_path, 'infor_test_dice_20subj_res_cond.txt')
file_loss_test = open(test_loss_path, 'a')
file_dice_test = open(test_dice_path, 'a')  
for i, data in enumerate(testloader):
    test_input_path_name = dataset_test.samples[i]
    file_name = ntpath.basename(test_input_path_name)
    under_score = file_name.rfind('_')
    subj_name = file_name[:22]    
    y_inputs, y_target = data[0].to(device), data[1].to(device)   
    print('#'*10, 'Sample: ', i, '#'*10)
    print(y_inputs.size())
    print(y_target.size())
    pd_out = net(y_inputs.float())
    # save the predict data
    move_cpu = pd_out.cpu()
    pd_out_save = move_cpu.detach().numpy()
    save_pred_path = os.path.join(test_path, 'normalized_brain_pred')
    save_subj_name = os.path.join(save_pred_path, ('%s_E_pd.mat' % (subj_name)))
    # save it as mat
    scipy.io.savemat(save_subj_name, dict(pred=pd_out_save))
    # cal loss
    pd_loss = criterion(pd_out, y_target.float())
    print('Loss: ', pd_loss.item())
    test_loss_record = {'subj_name': subj_name, 'pd_loss': pd_loss}     
    file_loss_test.write(repr(test_loss_record) + '\n') 
    # cal dice
    dice_test = Dice.cal_dice_one_batch(y_inputs.cpu(), y_target.cpu(), pd_out.cpu(), batch_size=1)
    print('[INFO]: --Dice-- ', dice_test)   
    test_dice_record = {'subj_name': subj_name, 'dice_loss': dice_test}     
    file_dice_test.write(repr(test_dice_record) + '\n')     
    
    
    total_loss += pd_loss.item()
test_mean_loss =  total_loss / len(testloader)  
print("Mean loss:", total_loss / (i+1)) 
file_loss_test.close()
file_dice_test.close()


## post-precessing: save the related information in training and testing
# write revalant information to a text file
#infor_record = {'used_time': time_elapsed, 'train_mean_loss': train_mean_loss,
#                'test_mean_loss':test_mean_loss}
#f = open('infor_recored.txt', 'w')
#f.write('dict = ' + repr(infor_record) + '\n')
#f.close()
print('#'*10, 'Done', '#'*10)