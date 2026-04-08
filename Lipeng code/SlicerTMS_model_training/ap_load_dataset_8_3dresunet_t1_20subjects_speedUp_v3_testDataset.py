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
test_path = os.path.join(root_path, 'testing_data_t1_temp') # /media/NAS/NAS/tms_data/testing_data
model_save_path = os.path.join('/rfanfs/pnl-zorro/home/gp88/software/anaconda3/doc/guoping_test/proj1_doc/model_save_subj20_t1')
print('model_saved_path:', model_save_path)
  

## loada data
batch_size = 2
#dataset_train = LD.RandomSamples(train_path, loader=LD.nifti_numpy_loader, 
#                                 loader_dataset=LD.make_dataset_subject)
#train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=4,
#                              shuffle=True)
## load a subject, check a subject
## Note: here the batch is dict
#sample, target = next(iter(train_loader))
#print("Sample size: ", sample.size())
#print("Target size:", target.size())
#print("[INFO]: Total number of trainings samples", len(train_loader) * batch_size)


## build network
# Loading the model
in_channels = 4
out_channels = 3
base_n_filter = 16
initial_lr = 0.002
net = Modified3DUNet(in_channels, out_channels, base_n_filter) 
net = net.float()
# select gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
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

#import torch.nn.functional as F
## another wey to define loss
#def cal_mask_loss(output, target, mask):
#    output = output * mask
#    loss = F.mse_loss(output, target, reduction='mean')
#    return loss

## training model function

#file_loss = open('Log_Result/infor_train_loss_45subj_res_cond.txt', 'a')
#def train_model(trainloader, net, criterion, optimizer, 
#                scheduler, num_epochs=25, batch_size=4):
#    time_started = time.time()  
#    num_steps = 0
#    # save log
#    writer = SummaryWriter(log_dir="log_45subj_cond_res")   
#    num_steps = 0
#    train_loss_total = 0.0
#    running_loss = 0.0 
#    for epoch in tqdm(range(num_epochs)):
#        start_time_epoch = time.time()
#        scheduler.step()
#        lr = scheduler.get_lr()[0]
#        writer.add_scalar('learning_rate', lr, epoch)          
#        
#        for i, data in enumerate(trainloader, 0): # index = 0
#            # load data
#            inputs, labels= data[0].to(device), data[1].to(device)
#            
#            # one iteration time
#            one_it_time_beg = time.time()
#            
#            # forward
#            outputs = net(inputs.float())
#            loss = criterion(outputs, labels.float())
#            # zero the params gradients
#            optimizer.zero_grad()            
#            loss.backward()
#            optimizer.step()
#            
#            running_loss += loss.item()
#            train_loss_total += loss.item()
#            num_steps += 1
#            # write infor to file
#            train_loss_record = {'iter_number': num_steps, 'train_loss': running_loss/batch_size}
#            file_loss.write(repr(train_loss_record) + '\n')
#            
#            if (i+1) % 2 == 0:    # print every 2000 mini-batches
#                print('[Epoch: %d/%d, Batch index: %5d] loss: %.3f' %
#                      (epoch + 1, num_epochs, i + 1, running_loss/batch_size))
##                train_loss_total_avg = train_loss_total / num_steps  
#                writer.add_scalars('losses', {'train_loss': running_loss/batch_size}, num_steps) 
#            # update running_loss
#            running_loss = 0.0       
#            # display training loss after 100 iterations
#            if num_steps % 1000 == 0:
#                print('='*30)
#                mean_loss_after_period = train_loss_total / (1000.0 * batch_size)
#                print('[INFO]: The Training Loss is %.3f in iteration %d' % (mean_loss_after_period, num_steps))
#                # update the total loss
#                train_loss_total = 0.0
#            
#            one_it_time_end = time.time() - one_it_time_beg
#            print('Used time in a iteration: %.2f' % (one_it_time_end))
#                        
#            
#        # check time
#        end_time_epoch = time.time()
#        total_time_epoch = end_time_epoch - start_time_epoch
#        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch+1, total_time_epoch))      
#        # save the model after 10 epochs: Epoch from 0 to end or the last epoch
#        if (epoch+1) % 1 == 0 or epoch == (num_epochs-1):
#            print('[INFO]:Epoch---saving model', epoch + 1)
#            print('[INFO]: Time Elapsed: %.2f' % (total_time_epoch))
#            model_save_name = "subj45_cond_res_%d.pt" % (epoch + 1)
#            model_save = os.path.join(model_save_path, model_save_name)
#            torch.save(net, model_save)
#            
#    print('Finished Training') 
#    train_mean_loss = train_loss_total / (len(trainloader)*(num_epochs))
#    print('Mean Loss: ', train_mean_loss)    
#    print('#' * 20)
#    time_elapsed = time.time() - time_started
#    print('Training complete in {:.0f}m {:.0f}s'.format(
#            time_elapsed // 60, time_elapsed % 60))
#    print('#' * 20)
#    # close file
#    file_loss.close()
#          
#    return net, time_elapsed, train_mean_loss
#    
### do training  
## define loss func and optim
#num_epochs = 30
#initial_lr = 0.004
##criterion = nn.MSELoss()
##optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
##optimizer = optim.Adam(net.parameters(),lr=initial_lr, betas=(0.9, 0.999), eps=1e-08)
## Use RAdam as Optim
#optimizer = RD.RAdam(net.parameters(),lr=initial_lr, betas=(0.9, 0.999), eps=1e-08)
## set learning_rate strategry: Decay LR by a factor of 0.1 every 10 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
##exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
#criterion = nn.MSELoss()
#
### train and save the model path
#print('#'*10, 'Training', '#'*10)
#net, time_elapsed, train_mean_loss = train_model(train_loader, net, criterion, 
#        optimizer, exp_lr_scheduler, num_epochs=num_epochs, batch_size=batch_size)      

## testing
# load model
model_path = os.path.join(model_save_path, 'model_subj20_t1_res_epoch_23.pth.tar')
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

dataset_test = LD.RandomSamples(test_path, loader=LD.nifti_numpy_loader, 
                                 loader_dataset=LD.make_dataset_subject_T1)
testloader = DataLoader(dataset=dataset_test,
                         batch_size=1, shuffle=False) 
print("[INFO]: Total number of testing samples", len(testloader))
total_loss = 0.0
test_loss_path = os.path.join(test_path, 'infor_test_loss_20subj_res_t1.txt')
test_dice_path = os.path.join(test_path, 'infor_test_dice_20subj_res_t1.txt')
file_loss_test = open(test_loss_path, 'a')
file_dice_test = open(test_dice_path, 'a')  
for i, data in enumerate(testloader):
    test_input_path_name = dataset_test.samples[i]
    file_name = ntpath.basename(test_input_path_name)
    under_score = file_name.rfind('_')
    subj_name = file_name[:15]    
    y_inputs, y_target = data[0].to(device), data[1].to(device)   
    print('#'*10, 'Sample: ', i, '#'*10)
    print(y_inputs.size())
    print(y_target.size())
    pd_out = net(y_inputs.float())
    # save the predict data
    move_cpu = pd_out.cpu()
    pd_out_save = move_cpu.detach().numpy()
    save_pred_path = os.path.join(test_path, 'simu_t1_pd_20')
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