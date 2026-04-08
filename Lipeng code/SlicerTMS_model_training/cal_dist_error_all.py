#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:08:20 2019

@author: gx020
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as pl
import scipy.io
#import sys
#sys.path.insert(0, "/rfanfs/pnl-zorro/home/gp88/software/anaconda3/doc/guoping_test/proj1_doc")
#from Nifti_Load_Tensor import RandomSamples, nifti_numpy_loader
#from Load_Data import LoadData as LD
import glob 

# show one slice
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[i].axis('off')

#root_path = '/run/media/ln915/Elements/TMSExperiment/ernie/Simulation/Magstim/ernie_simu_cond/testing/'
#test_name = '100408/100408_TMS_0200_Ds.nii.gz'
#test_label_gt = '100408/100408_TMS_0200_E.nii.gz'
#test_label_pd = 'SurfaceTesting/simu_cond_pd_20/100408_pred/100408_TMS_0020_E_pd.mat'
#test_subj = os.path.join(root_path, test_name)
#test_gt = os.path.join(root_path, test_label_gt)
#test_pd = os.path.join(root_path, test_label_pd)

# load files    
# load data : confirm the pd and gt has the same idx
# read the traing samples's dir and simulation's dir
subj_name = 'subj_10000'   
def obtain_gt_pd_filename(root_path,subj_name):
    gt_path = os.path.join(root_path, subj_name, '*_E.nii.gz')
    pd_path_tmp = ('prediction/%s_pred/*.mat' % (subj_name))
    pd_path = os.path.join(root_path, pd_path_tmp)
    print(gt_path)
    print(pd_path)
    gt_dir = [f for f in glob.glob(gt_path)]
    pd_dir = [f for f in glob.glob(pd_path)]
    # make sure they are in the same order
    gt_dir.sort()
    pd_dir.sort()
    print(gt_dir)
    # check the samples, targets and masks are in the same order
    for sample, target in zip(gt_dir, pd_dir):
        last_underscore_gt = sample.rfind('_')
        last_slash_gt = sample.rfind('/')
        sample_id = sample[last_slash_gt:last_underscore_gt]
        last_underscore_pd = target.rfind('_')
        last_slash_pd = target.rfind('/')
        targe_id = target[last_slash_pd:last_underscore_pd-2]
        assert(sample_id == targe_id)

    return gt_dir, pd_dir
    
    
def load_gt_pd(gt_dir, pd_dir):
    # load gt E filed and gt E field
    gt_file = (nib.load(gt_dir)).get_fdata() # (180, 220, 120, 3)
    pd_file = scipy.io.loadmat(pd_dir)['pred'].transpose(2, 3, 4, 1, 0) # 180*220*120*3*1
    pd_file = np.squeeze(pd_file, axis=(4))# 180*220*120*3
    
    return gt_file, pd_file

# 1 get the brain mask
# cal mask
#file_dir = '/media/NAS/NAS/tms_data/testing_data/162733/162733_TMS_0200_Ds.nii.gz'   
#save_mask_path = '/rfanfs/pnl-zorro/home/gp88/software/anaconda3/doc/guoping_test/proj1_doc/Analysis-Testing-Dataset' 
#subj_mask_name = '162733_WM_Mask'
#mask_path_name = os.path.join(save_mask_path, ('%s.mat' % (subj_mask_name)))

def obtain_brain_mask(file_dir, mask_path_name):
    # load Ds
    f_cond_1 = (nib.load(file_dir)).get_fdata() 
    f_cond = f_cond_1[:,:, :, :3]
#    print(f_cond.shape)
    # Norm
    f_norm = np.sqrt(np.sum(np.power(f_cond, 2), axis=3))
    
    # threshold--- mask
    #% set skul, csf,etc to 0
    #D(D<0.011) = 0;ƒ
    #D(D>1.65) = 0;
    #D(D>0.465&D<0.466) = 0;
    #ID_brain = D>0;
    f_norm[(f_norm < 0.011)]=0
    # or (f_norm > 1.65) or (f_norm > 0.465 and f_norm < 0.466)
    f_norm[(f_norm > 1.65)]=0
    f_norm[np.logical_and((f_norm >= 0.465),(f_norm < 0.466))]=0
    brain_mask = (f_norm > 0)
    # save it as mat
#    scipy.io.savemat(mask_path_name, dict(mask=brain_mask))    
    return brain_mask, f_norm



# cal distance eroor between two maximaum E value and loss percent in the brain
#mask_dir = ('%s_WM_Mask.mat' % (subj_name))
#brain_mask = scipy.io.loadmat(mask_dir)['mask'] # 180*220*120
def cal_dist_loss(brain_mask, gt_dir, pd_dir):
    gt_file, pd_file = load_gt_pd(gt_dir, pd_dir)
    # norm the E vector
    norm_gt = np.sqrt(np.sum(np.power(gt_file, 2), axis=3))
    brain_gt = norm_gt * brain_mask
    norm_pd = np.sqrt(np.sum(np.power(pd_file, 2), axis=3))
    brain_pd = norm_pd * brain_mask
    # find the coodinate of the maximum E 
    gt_idx = np.unravel_index(np.argmax(brain_gt), brain_gt.shape, order='C')
    pd_idx = np.unravel_index(np.argmax(brain_pd), brain_pd.shape, order='C') 
    
    # cal dis error
    error_power = np.power((np.array(gt_idx) - np.array(pd_idx)), 2)
    dist_error = np.sqrt(np.sum(error_power))
    # cal Absolute loss percent in the brain
    loss_volume_abs = np.abs(brain_gt - brain_pd)
    loss_volume_per = np.sum((loss_volume_abs / (brain_gt+1e-9))) / np.sum(brain_mask)
    # cal similarilty: cos(theta)
    simu_numerator = norm_gt * norm_pd + 1e-6
    simi_all = ((gt_file * pd_file) * np.expand_dims(brain_mask, axis=3)) / np.expand_dims(simu_numerator, axis=3)
    simu_mean = np.sum(simi_all) / np.sum(brain_mask)
    
    return dist_error, loss_volume_per, simu_mean

# cal dice
# cal effective simulation region
def cal_simu_mask_speed(simu_brain_norm, brain_mask, percent):
    # norm the simu vector 
    simu_mask = np.greater_equal(simu_brain_norm, 
                                   np.percentile(simu_brain_norm[brain_mask], percent))
    return simu_mask    

def cal_simu_mask(simu_result, brain_mask, percent):
    # norm the simu vector 
#    print('simu_result', simu_result.shape)
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
    
#    denominator = (label_gt + label_pd).sum(-1) * 2 ## label_gt[1] = False, logical : this is wrong
    denominator = (label_gt + label_pd).sum(-1) + intersect # Or: denominator = label_gt.sum() + label_pd.sum()    
    simu_dice = (2. * intersect) / denominator.clip(0.0001, denominator.max()) 
    return simu_dice

# add correlate coefficient and mean absolute error
def cal_dist_loss_cor_more(brain_mask, gt_dir, pd_dir):
    brain_mask = brain_mask > 0 # change to logic type
    gt_file, pd_file = load_gt_pd(gt_dir, pd_dir)
    # norm the E vector
    norm_gt = np.sqrt(np.sum(np.power(gt_file, 2), axis=3))
    brain_gt = norm_gt * brain_mask
    norm_pd = np.sqrt(np.sum(np.power(pd_file, 2), axis=3))
    brain_pd = norm_pd * brain_mask
    # find the coodinate of the maximum E 
    gt_idx = np.unravel_index(np.argmax(brain_gt), brain_gt.shape, order='C')
    pd_idx = np.unravel_index(np.argmax(brain_pd), brain_pd.shape, order='C') 
    
    # cal dis error
    error_power = np.power((np.array(gt_idx) - np.array(pd_idx)), 2)
    dist_error = np.sqrt(np.sum(error_power))
    # cal Absolute loss percent in the brain
    loss_volume_abs = np.abs(brain_gt - brain_pd)
    loss_volume_per = np.sum((loss_volume_abs / (brain_gt+1e-9))) / np.sum(brain_mask)
    # cal similarilty: cos(theta)
    simu_numerator = norm_gt * norm_pd + 1e-6
    simi_all = ((gt_file * pd_file) * np.expand_dims(brain_mask, axis=3)) / np.expand_dims(simu_numerator, axis=3)
    simu_mean = np.sum(simi_all) / np.sum(brain_mask)
    
    # cal correlate coefficient in the brain
    norm_element_gt = np.extract(brain_mask, norm_gt)
    norm_element_pd = np.extract(brain_mask, norm_pd)
    cor_num = np.corrcoef(norm_element_gt, norm_element_pd)[0, 1]
    # cal MAE in brain
    mae_brain = np.sum(loss_volume_abs) / np.sum(brain_mask)
    
    # cal dice
#    print('brain_gt', brain_gt.shape)
#    print('brain_mask', brain_mask.shape)
    simu_mask_gt = cal_simu_mask_speed(brain_gt, brain_mask, 95)
    simu_mask_pd = cal_simu_mask_speed(brain_pd, brain_mask, 95)
    
#    simu_mask_gt = cal_simu_mask(gt_file, brain_mask, 95)
#    simu_mask_pd = cal_simu_mask(pd_file, brain_mask, 95)    
    simu_dice= cal_dice(simu_mask_gt, simu_mask_pd)    
    
    
    return dist_error, loss_volume_per, simu_mean, cor_num, mae_brain, simu_dice

if __name__ == "__main__":
    root_path = './'
    save_path = './measure_result'
    subj_names = [ 'subj_10000']  #  '162733',  '163129', '162733', '100408', '101107',
    
#    # obtain mask
#    file_dir = '/media/NAS/NAS/tms_data/testing_data/163129/163129_TMS_0200_Ds.nii.gz'   
#    save_mask_path = '/rfanfs/pnl-zorro/home/gp88/software/anaconda3/doc/guoping_test/proj1_doc/Analysis-Testing-Dataset' 
#    subj_mask_name = '163129_WM_Mask'
#    mask_path_name = os.path.join(save_mask_path, ('%s.mat' % (subj_mask_name)))
#    obtain_brain_mask(file_dir, mask_path_name)
#    print('done')
    
    for subj_name in subj_names:
        print(subj_name)
        # save file name
        sample_name_path = ('%s_subj20_cond_sample_name.txt' % (subj_name))
        file_sample_name = open(os.path.join(save_path, sample_name_path), 'a')
        dist_err_path= ('%s_subj20_cond_dist_error.txt' % (subj_name))
        file_dist_err = open(os.path.join(save_path, dist_err_path), 'a')
        loss_path= ('%s_subj20_cond_loss_percent.txt' % (subj_name))
        file_loss_per = open(os.path.join(save_path, loss_path), 'a')  
        simu_path= ('%s_subj20_cond_simu_mean.txt' % (subj_name))
        file_simu_mean = open(os.path.join(save_path, simu_path), 'a')   
        
        corr_path= ('%s_subj20ƒ_cond_corr_num.txt' % (subj_name))
        file_corr_num = open(os.path.join(save_path, corr_path), 'a') 
        mae_path= ('%s_subj20_cond_mae_mean.txt' % (subj_name))
        file_mae_mean = open(os.path.join(save_path, mae_path), 'a') 
        dice_path= ('%s_subj20_cond_dice_all.txt' % (subj_name))
        file_dice_all = open(os.path.join(save_path, dice_path), 'a')         
        
        # load mask
        mask_dir = ('%s_GM_Mask.mat' % (subj_name))
	
        brain_mask = scipy.io.loadmat(mask_dir)['mask'] # 180*220*120
        # load gt and pd filenames
	
        gt_dirs, pd_dirs = obtain_gt_pd_filename(root_path,subj_name)

        # load gt and pd file and # cal dist error, loss percent and similarity
        for gt_dir, pd_dir in zip(gt_dirs, pd_dirs):      
            print(gt_dir)
#            dist_error, loss_volume_per, simu_mean = cal_dist_loss(brain_mask, gt_dir, pd_dir)
            dist_error, loss_volume_per, simu_mean, cor_num, mae_brain, simu_dice = cal_dist_loss_cor_more(brain_mask, gt_dir, pd_dir)
            print((dist_error, loss_volume_per, simu_mean, cor_num, mae_brain, simu_dice))
            # save the result to files  
            file_sample_name.write(str(gt_dir) + '\n')               
            file_dist_err.write(str(dist_error) + '\n')               
            file_loss_per.write(str(loss_volume_per) + '\n')                
            file_simu_mean.write(str(simu_mean) + '\n') 
            
            file_corr_num.write(str(cor_num) + '\n')                
            file_mae_mean.write(str(mae_brain) + '\n')  
            file_dice_all.write(str(simu_dice) + '\n')  
        # close file
        file_sample_name.close()
        file_dist_err.close()
        file_loss_per.close()  
        file_simu_mean.close()
        file_corr_num.close()  
        file_mae_mean.close() 
        file_dice_all.close()
        print('[INFO]: Complete ', subj_name)
            
        
        
    
    
