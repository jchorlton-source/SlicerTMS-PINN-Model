#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:27:13 2019

@author: gx020
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
# show one slice
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[i].axis('off')

# cal mask
def obtain_brain_mask(f_cond):
    # Norm
    f_norm = np.sqrt(np.sum(np.power(f_cond, 2), axis=3))
    
    # threshold--- mask
    #% set skul, csf,etc to 0
    #D(D<0.011) = 0;
    #D(D>1.65) = 0;
    #D(D>0.465&D<0.466) = 0;
    #ID_brain = D>0;
    f_norm[(f_norm < 0.011)]=0
    # or (f_norm > 1.65) or (f_norm > 0.465 and f_norm < 0.466)
    f_norm[(f_norm > 1.65)]=0
    f_norm[np.logical_and((f_norm >= 0.465),(f_norm < 0.466))]=0
    brain_mask = (f_norm > 0)
    
    return brain_mask, f_norm

# cal effective simulation region
def cal_simu_mask(simu_result, brain_mask, percent):
    # norm the simu vector 
    simu_norm = np.sqrt(np.sum(np.power(simu_result, 2), axis=3))
    simu_brain = simu_norm * brain_mask
    simu_mask = np.greater_equal(simu_brain, 
                                   np.percentile(simu_brain[brain_mask], percent))
    return simu_mask

# cal dice
def cal_dice(gt_mask, pd_mask):
    label_gt = gt_mask.flatten()
    label_pd = pd_mask.flatten()
    
    # Compute per channel Dice Coefficient
    intersect = (label_gt * label_pd).sum(-1)
    
    denominator = (label_gt + label_pd).sum(-1) + intersect # Or: denominator = label_gt.sum() + label_pd.sum()
    simu_dice = (2. * intersect) / denominator.clip(0.0001, denominator.max()) 
    return simu_dice
            
def preprocess_tensor_numpy(input_sample, simu_gt_vect, simu_pd_vect):
    # change dimension
    input_sample = input_sample.detach()
    simu_gt_vect = simu_gt_vect.detach()
    simu_pd_vect = simu_pd_vect.detach()
    one_cond_sample = input_sample[:3, :, :, :].numpy().transpose(1, 2, 3, 0)
    one_simu_gt = simu_gt_vect[:, :, :, :].numpy().transpose(1, 2, 3, 0)
    one_simu_pd = simu_pd_vect[:, :, :, :].numpy().transpose(1, 2, 3, 0)
    return one_cond_sample, one_simu_gt, one_simu_pd

def cal_dice_one_batch(batch_input_sample, 
                       batch_simu_gt_vector, batch_simu_pd_vect, batch_size):
    simu_dice = np.zeros((1, batch_size))
    for num in range(batch_size):
        input_one_sample = batch_input_sample[num, :, :, :, :]
        one_simu_gt_vect = batch_simu_gt_vector[num, :, :, :, :]
        one_simu_pd_vect = batch_simu_pd_vect[num, :, :, :, :]
        one_cond_sample, one_simu_gt, one_simu_pd = preprocess_tensor_numpy(input_one_sample, 
                                                                            one_simu_gt_vect, one_simu_pd_vect)
        brain_mask, f_norm = obtain_brain_mask(one_cond_sample)
        simu_mask_gt = cal_simu_mask(one_simu_gt, brain_mask, 95)
        simu_mask_pd = cal_simu_mask(one_simu_pd, brain_mask, 95)
        simu_dice[:, num]= cal_dice(simu_mask_gt, simu_mask_pd)
        
    return simu_dice        
        
        
        
        
        
        

    
#if __name__ == "__main__":
#    dice_all = cal_dice_one_batch(subj, simu, simu, 2)
#    # load data
#    # subj size: torch.Size([2, 6, 180, 220, 120])
#    batch_cond = subj[:, :3, :, :, :].numpy().transpose(2, 3, 4, 1, 0)
#    one_batch_cond = batch_cond[:, :, :, :, 1]
#    brain_mask, cond_norm = obtain_brain_mask(one_batch_cond)
#    
#    batch_simu= simu[:, :, :, :, :].numpy().transpose(2, 3, 4, 1, 0)
#    one_batch_simu = batch_simu[:, :, :, :, 1]
#    simu_mask_gt = cal_simu_mask(one_batch_simu, brain_mask, 95)
#    
#    dice_simu = cal_dice(brain_mask, simu_mask_gt)
#    
#    # show one slice
#    slice_0 = cond_norm[:, :, 30]
#    slice_1 = brain_mask[:, :, 30]
#    slice_2 = simu_mask_gt[:, :, 30]
#    show_slices([slice_0, slice_1, slice_2])
#    plt.suptitle("Center slices for Cond image")  # doctest: +SKIP





