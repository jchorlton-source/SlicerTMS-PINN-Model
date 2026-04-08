#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:07:38 2019

@author: jeg88
"""

import os
import argparse
import datetime
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
#from Nifti_Load_Tensor import RandomSamples, nifti_numpy_loader
from Load_Data import LoadData as LD
from Loss_Function import LossFun as LF
from Loss_Function import CalDice as Dice
import Train_Data_Loader as TD
#from medicaltorch import transforms as mt_transforms
#from medicaltorch import datasets as mt_datasets
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils
from Optim_Method import radam as RD
# FP32-FP16
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
from apex.parallel import DistributedDataParallel 
import shutil 

os.environ['RANK'] == '0'

## root setting
root_path = '/run/media/gx020/TMSData'
train_path = os.path.join(root_path, 'training_data')
test_path = os.path.join(root_path, 'testing_data')
val_path = os.path.join(root_path, 'validation_data')
model_save_path = os.path.join('/rfanfs/pnl-zorro/home/gp88/software/anaconda3/doc/guoping_test/proj1_doc/model_save_subj20_t1')
print('model_saved_path:', model_save_path)

# select gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# put the model on GPUs by model.to(device)
parser = argparse.ArgumentParser()
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--global_rank', default=0, type=int,
                        help='Used for multi-machine training.')
args = parser.parse_args()

# FOR DISTRIBUTED:  If we are running under torch.distributed.launch,
# the 'WORLD_SIZE' environment variable will also be set automatically.
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    print("Using Parallel-------------------1")
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    print("Using Parallel-------------------2")
    print("Args.local_rank: %d, args.global_rank: %d" % (args.local_rank, args.global_rank))
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    
torch.backends.cudnn.benchmark = True

print('[INFO]: load data---------------')

## load data
batch_size = 2
dataset_train = LD.RandomSamples(train_path, loader=LD.nifti_numpy_loader, 
                                 loader_dataset=LD.make_dataset_subject_T1)
#dataset_val = LD.RandomSamples(val_path, loader=LD.nifti_numpy_loader, 
#                                 loader_dataset=LD.make_dataset_subject_T1)
train_sampler = None
val_sampler = None
if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
#        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val) 
train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size= batch_size, shuffle=(train_sampler is None),
        num_workers=8, pin_memory=True, sampler=train_sampler)
print("[INFO]: Total number of trainings samples for each GPU", len(train_loader) * batch_size)

#val_loader = torch.utils.data.DataLoader(
#        dataset_val, batch_size = 1, shuffle=(val_sampler is None),
#        num_workers=8, pin_memory=True, sampler=val_sampler)       

## build network
# Loading the model
in_channels = 4
out_channels = 3
base_n_filter = 16
# define loss func and optim
num_epochs = 25
initial_lr = 0.002
net = Modified3DUNet(in_channels, out_channels, base_n_filter).cuda()
#criterion = nn.MSELoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(net.parameters(),lr=initial_lr, betas=(0.9, 0.999), eps=1e-08)
# Use RAdam as Optim
optimizer = RD.RAdam(net.parameters(),lr=initial_lr, betas=(0.9, 0.999), eps=1e-08)
# set learning_rate strategry: Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
#exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
criterion = nn.MSELoss().cuda()
####

# Allow Amp to perform casts as required by the opt_level
net, optimizer = amp.initialize(net, optimizer, opt_level="O2")
if args.distributed:
    print("Using Parallel-------------------3")
    # FOR DISTRIBUTED:  After amp.initialize, wrap the model with
    # apex.parallel.DistributedDataParallel.
#    net = DistributedDataParallel(net, delay_allreduce=True)
    # torch.nn.parallel.DistributedDataParallel is also fine, with some added args:
    net = torch.nn.parallel.DistributedDataParallel(net, 
                                                   device_ids=[args.local_rank],
                                                   output_device=args.local_rank, find_unused_parameters=True)  

## training model function
def train_model(trainloader, net, criterion, optimizer, 
                scheduler, num_epochs=25, batch_size=2):
    file_loss_train = open('model_save_subj20_t1/infor_train_loss_20subj_res_cond_next24.txt', 'a')
#    file_loss_val = open('model_save_subj20_t1/infor_val_loss_20subj_res_cond.txt', 'a') 
    file_epoch_time = open('model_save_subj20_t1/epoch_time_next24.txt', 'a')  
    # record time
    # write starting time
    currentDT = datetime.datetime.now()
    begin_time = {'start time': str(currentDT)}                
    file_loss_train.write(repr(begin_time) + '\n') 
    
    time_started = time.time()  
    num_steps = 0
    # save log
    writer = SummaryWriter(log_dir="log_20subj_t1_resunet_continue")   
    num_steps = 0
    train_loss_total = 0.0
    running_loss = 0.0 
    losses_save = []    
    for epoch in tqdm(range(num_epochs)):
        start_time_epoch = time.time()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch) 
        flag_dist = torch.distributed.get_rank()        
                
        for i, data in enumerate(trainloader, 0): # index = 0
            inputs, labels= data[0].to(device), data[1].to(device)
#            # one iteration time
#            one_it_time_beg = time.time()
            
            # forward
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.float())
            losses_save.append(loss.item())
            # zero the params gradients
            optimizer.zero_grad()    
            
            # when doing .backward, let amp do it so it can scale the loss
            with amp.scale_loss(loss, optimizer) as scaled_loss:                      
                scaled_loss.backward()                       
#            loss.backward() # need long time to do this step
            # update
            optimizer.step() 
            # complete a iter                     
#            one_it_time_end = time.time() - one_it_time_beg
#            print('Used time in %d iterations: %.2f\n' % (i+1, one_it_time_end))
            
            # cal loss
            running_loss += loss.item()
            train_loss_total += loss.item()            
            num_steps += 1 # total steps
            # show infor in terminal and record loss and dice
            if (i+1) % 10 == 0:    # print every 2 mini-batches
                print('[Epoch: %d/%d, Batch index: %5d] loss: %.3f' %
                      (epoch + 1, num_epochs, i + 1, running_loss))
#                train_loss_total_avg = train_loss_total / num_steps  
#                # cal dice
#                dice_train = Dice.cal_dice_one_batch(inputs.cpu(), labels.cpu(), outputs.cpu(), batch_size)
#                print('[INFO]: --Dice-- ', dice_train)
#                train_dice_mean = dice_train.mean()
                # write it to tensorboard
                writer.add_scalars('losses', {'train_loss': running_loss}, num_steps)
#                writer.add_scalars('dice', {'train_dice_mean': train_dice_mean}, num_steps)  
                # write infor to file
                train_loss_record = {'iter_number': num_steps, 'train_loss': losses_save}                
                file_loss_train.write(repr(train_loss_record) + '\n') 
                # update list
                losses_save = []
            # update running_loss
            running_loss = 0.0       
            # display training loss after 1000 iterations and cal val_loss and val_dice
            if num_steps % 1000 == 0 and args.local_rank == 0:
                print('='*30)
                mean_loss_after_period = train_loss_total / (1000.0)
                print('[INFO]: The Training Loss is %.3f in iteration %d' % (mean_loss_after_period, num_steps))
                # update the total loss
                train_loss_total = 0.0               
#                # do validation
#                # load validation data
#                total_loss_val = []
#                total_dice_val = []
#                for val_num, val_data in enumerate(val_loader, 0): # index = 0
#                    inputs_val, labels_val = val_data[0].to(device), val_data[1].to(device)
#                    if val_num < 1: # selcet 20 sample for validation each time
#                        with torch.no_grad():
#                            output_val = net(inputs_val.float())
#                            loss_val = criterion(output_val, labels_val.float())
#                            total_loss_val.append(loss_val.item())
#                            dice_val = Dice.cal_dice_one_batch(inputs.cpu(), labels.cpu(), outputs.cpu(), 1)  
#                            total_dice_val.append(dice_val)
#                    else:                       
#                        val_mean = np.array(total_loss_val).mean()
#                        print('[INFO]: Validation mean loss: %.2f' % (val_mean))
#                        val_dice_mean = np.array(total_dice_val).mean()
#                        # write infor to file
#                        val_loss_record = {'iter_number': num_steps, 'val_loss': val_mean,
#                                             'val_dice_mean': val_dice_mean}                
#                        file_loss_val.write(repr(val_loss_record) + '\n')                        
#                        # update total_val_loss
#                        total_loss_val = []
#                        # write it to tensorboard
#                        writer.add_scalars('losses', {'val_loss': val_mean}, num_steps) 
#                        writer.add_scalars('dice', {'val_dice_mean': val_dice_mean}, num_steps)
#                        break
                    
            # save model after 5000 iters
            if num_steps % 5000 == 0 and args.local_rank == 0:
                model_save_name = "model_subj20_t1_res_iter_%d.pth" % (num_steps)
                model_save = os.path.join(model_save_path, model_save_name)
                # save in other parameters
                torch.save(net.state_dict(), model_save)   
                model_save_name = "optimizer_subj20_t1_res_iter_%d.pth" % (num_steps)
                model_save = os.path.join(model_save_path, model_save_name)
                # save optimizer
                torch.save(optimizer.state_dict(), model_save)                 
                    
        ## check time and save model after a epoch
        end_time_epoch = time.time()
        total_time_epoch = end_time_epoch - start_time_epoch
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch+1, total_time_epoch))      
        # write each epoch time to file
        epoch_time_record = {'Epoch': epoch+1, 'spend_time': total_time_epoch,
                             'Total_Iter_Num_run': num_steps}                
        file_epoch_time.write(repr(epoch_time_record) + '\n')
        # save the model after 1 epochs: Epoch from 0 to end or the last epoch
        if (epoch+1) % 1 == 0 or epoch == (num_epochs-1):
            if args.local_rank == 0 and flag_dist == 0:
                print('[INFO]:Epoch---saving model', epoch + 1)
                print('[INFO]: Time Elapsed: %.2f' % (total_time_epoch))
                model_save_name = "model_subj20_t1_res_epoch_%d.pth.tar" % (epoch + 1)
                model_save = os.path.join(model_save_path, model_save_name)
                torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Iter_Num': num_steps,
                'loss':loss
                }, model_save)
            
    print('Finished Training')  
    print('#' * 20)
    time_elapsed = time.time() - time_started
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    print('#' * 20)
          
    # write end time
    currentDT = datetime.datetime.now()
    end_time = {'start time': str(currentDT)}                
    file_loss_train.write(repr(end_time) + '\n')          
    # close file
    file_loss_train.close()
#    file_loss_val.close()
    file_epoch_time.close()
          
    return net, time_elapsed, train_mean_loss
    
####
## do training  
## train and save the model path
print('#'*10, 'Training', '#'*10)
net, time_elapsed, train_mean_loss = train_model(train_loader, net, criterion, 
        optimizer, exp_lr_scheduler, num_epochs=num_epochs, batch_size=batch_size)
print('#'*10, 'Training Completed', '#'*10)  

## post-precessing: save the related information in training and testing
# write revalant information to a text file
infor_record = {'used_time': time_elapsed}
f = open('model_save_subj20/infor_recored.txt', 'w')
f.write('dict = ' + repr(infor_record) + '\n')
f.close()
print('#'*10, 'Done', '#'*10)
