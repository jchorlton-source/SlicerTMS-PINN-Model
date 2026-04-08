#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:37:15 2019

@author: jeg88
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:31:45 2019

@author: jeg88
"""
import os
import numpy as np
import nibabel as nib
#from nibabel.testing import data_path
from torch.utils.data import Dataset
import glob
import os
import itertools

# load nifti data, transform it to numpy and change the axis order
def nifti_numpy_loader(file_full_dir):
#    print('Sample:', file_full_dir)
    temp = (nib.load(file_full_dir)).get_fdata() # change to narray
    if (temp.ndim == 4):
        entry = temp.transpose(3, 0, 1, 2) # CDHW
        entry[np.isnan(entry)] = 0
        entry[np.isinf(entry)] = 0
    else:
        entry = np.expand_dims(temp, axis=3)
        entry = entry.transpose(3, 0, 1, 2) # CDHW

    entry[np.isnan(entry)] = 0
    entry[np.isinf(entry)] = 0
    return entry


# read the traing samples's dir and simulation's dir
def make_dataset_one_folder_cond(root_path):
    sample_path = os.path.join(root_path, "*_Ds.nii.gz")
    target_path = os.path.join(root_path, "*_E.nii.gz")
    
    sample_dir = [f for f in glob.glob(sample_path)]
    target_dir = [f for f in glob.glob(target_path)]
    # make sure they are in the same order
    sample_dir.sort()
    target_dir.sort()
    
    # check the samples, targets and masks are in the same order
    for sample, target in zip(sample_dir, target_dir):
        #print('sample:', sample)
        
        last_underscore = sample.rfind('_')
        sample_id = sample[:last_underscore]
        target_id = target[:last_underscore]
        #print('sample id:', sample_id)
        #print('target id:', target_id)
        assert(sample_id == target_id)

    return sample_dir, target_dir


# read the traing samples's dir and simulation's dir
def make_dataset_one_folder_iso(root_path):
    sample_path = os.path.join(root_path, "*_DsI2A.nii.gz")
    target_path = os.path.join(root_path, "*_E.nii.gz")

    sample_dir = [f for f in glob.glob(sample_path)]
    target_dir = [f for f in glob.glob(target_path)]
    # make sure they are in the same order
    sample_dir.sort()
    target_dir.sort()

    # check the samples, targets and masks are in the same order
    for sample, target in zip(sample_dir, target_dir):
        print('sample:', sample)
        last_underscore = sample.rfind('_')
        sample_id = sample[:last_underscore]
        targe_id = target[:last_underscore]
        assert (sample_id == targe_id)

    return sample_dir, target_dir

def make_dataset_one_folder_std(root_path):
    sample_path = os.path.join(root_path, "*_DsStd.nii.gz")
    target_path = os.path.join(root_path, "*_E.nii.gz")

    sample_dir = [f for f in glob.glob(sample_path)]
    target_dir = [f for f in glob.glob(target_path)]
    # make sure they are in the same order
    sample_dir.sort()
    target_dir.sort()

    # check the samples, targets and masks are in the same order
    for sample, target in zip(sample_dir, target_dir):
        # print('sample:', sample)
        last_underscore = sample.rfind('_')
        sample_id = sample[:last_underscore]
        targe_id = target[:last_underscore]
        assert (sample_id == targe_id)

    return sample_dir, target_dir


# read the traing samples's dir and simulation's dir
def make_dataset_one_folder_T1D(root_path):
    sample_path = os.path.join(root_path, "*_Ds*")
    target_path = os.path.join(root_path, "*_E*")
    
    sample_dir = [f for f in glob.glob(sample_path)]
    target_dir = [f for f in glob.glob(target_path)]
    # make sure they are in the same order
    sample_dir.sort()
    target_dir.sort()
    
    # check the samples, targets and masks are in the same order
    for sample, target in zip(sample_dir, target_dir):
        last_underscore = sample.rfind('_')
        sample_id = sample[:last_underscore]
        target_id = target[:last_underscore]
        print('target id:', target_id)
        print('sample id:', sample_id)
        assert(sample_id == target_id)

    return sample_dir, target_dir

import pdb as pd 
# read the traing samples's dir and simulation's dir
def make_dataset_one_folder_T1D_v2(root_path_sample, root_path_gt):
    sample_path = os.path.join(root_path_sample, "*_Ds*")
    target_path = os.path.join(root_path_gt, "*_E*")
    
    sample_dir = [f for f in glob.glob(sample_path)]
    target_dir = [f for f in glob.glob(target_path)]
    # make sure they are in the same order
    sample_dir.sort()
    target_dir.sort()
    
    # check the samples, targets and masks are in the same order
    for sample, target in zip(sample_dir, target_dir):
#        print('sample:', sample)
#        print('gt:', target)
#        pd.set_trace()
        last_underscore = sample.rfind('_')
        sample_id = sample[last_underscore-4:last_underscore]
        last_underscore = target.rfind('_')
        targe_id = target[last_underscore-4:last_underscore]
        assert(sample_id == targe_id)

    return sample_dir, target_dir

# read the traing samples's dir and simulation's dir
# Suppose the samples, targets and masks are in the sample folder
def make_dataset_mask(root_path):
    sample_path = os.path.join(root_path, "*_Ds*")
    target_path = os.path.join(root_path, "*_E*")
    mask_path = os.path.join(root_path, "*_mask.nii.gz")
    
    sample_dir = [f for f in glob.glob(sample_path)]
    target_dir = [f for f in glob.glob(target_path)]
    mask_dir = [f for f in glob.glob(mask_path)]
    # make sure they are in the same order
    sample_dir.sort()
    target_dir.sort()
    mask_dir.sort()
    
    # check the samples, targets and masks are in the same order
    for sample, target, mask in zip(sample_dir, target_dir, mask_dir):
        last_underscore = sample.rfind('_')
        sample_id = sample[:last_underscore]
        targe_id = target[:last_underscore]
        mask_id = mask[:last_underscore]
        assert(sample_id == targe_id == mask_id)

    return sample_dir, target_dir, mask_dir  

 

# read all files from each subject folder
def make_dataset_subject(root_path):
    # root_path: includes all subjects files
    sample_dir = []
    target_dir = []
    sample_dir_temp, target_dir_temp = make_dataset_one_folder_cond(root_path)
    if (sample_dir_temp and target_dir_temp): # check whether it is empty
        sample_dir.append(sample_dir_temp)
        target_dir.append(target_dir_temp)

# #    subj_abs_path = [os.path.abspath(x) for x in os.listdir(root_path)]
#     subj_abs_path = [os.path.join(root_path, x) for x in os.listdir(root_path)]
#     for a_subj_folder in subj_abs_path:
#         sample_dir_temp, target_dir_temp = make_dataset_one_folder_cond(a_subj_folder)
#         if (sample_dir_temp and target_dir_temp): # check whether it is empty                  
#             sample_dir.append(sample_dir_temp)
#             target_dir.append(target_dir_temp)
    # obtain 1 dimension list
    sample_dir_all = list(itertools.chain(*sample_dir))
    target_dir_all = list(itertools.chain(*target_dir))
        
    return sample_dir_all, target_dir_all


def make_dataset_subject_iso(root_path):
    # root_path: includes all subjects files
    sample_dir = []
    target_dir = []

    #    subj_abs_path = [os.path.abspath(x) for x in os.listdir(root_path)]
    subj_abs_path = [os.path.join(root_path, x) for x in os.listdir(root_path)]
    for a_subj_folder in subj_abs_path:
        sample_dir_temp, target_dir_temp = make_dataset_one_folder_iso(a_subj_folder)
        if (sample_dir_temp and target_dir_temp):  # check whether it is empty
            sample_dir.append(sample_dir_temp)
            target_dir.append(target_dir_temp)
    # obtain 1 dimension list
    sample_dir_all = list(itertools.chain(*sample_dir))
    target_dir_all = list(itertools.chain(*target_dir))

    return sample_dir_all, target_dir_all

def make_dataset_subject_std(root_path):
    # root_path: includes all subjects files
    sample_dir = []
    target_dir = []
    sample_dir_temp, target_dir_temp = make_dataset_one_folder_std(root_path)
    if (sample_dir_temp and target_dir_temp): # check whether it is empty
        sample_dir.append(sample_dir_temp)
        target_dir.append(target_dir_temp)

    # obtain 1 dimension list
    sample_dir_all = list(itertools.chain(*sample_dir))
    target_dir_all = list(itertools.chain(*target_dir))
        
    return sample_dir_all, target_dir_all


# read all files from each subject folder
def make_dataset_subject_T1(root_path):
    # root_path: includes all subjects files
    sample_dir = []
    target_dir = []
#    subj_abs_path = [os.path.abspath(x) for x in os.listdir(root_path)]
    subj_abs_path = [os.path.join(root_path, x) for x in os.listdir(root_path)]
    for a_subj_folder in subj_abs_path:
        sample_dir_temp, target_dir_temp = make_dataset_one_folder_T1D(a_subj_folder)
        if (sample_dir_temp and target_dir_temp): # check whether it is empty                  
            sample_dir.append(sample_dir_temp)
            target_dir.append(target_dir_temp)
    # obtain 1 dimension list
    sample_dir_all = list(itertools.chain(*sample_dir))
    target_dir_all = list(itertools.chain(*target_dir))
        
    return sample_dir_all, target_dir_all  
   

#samples, targets = make_dataset(root_path, 'training', 'simu_field')
    
# dataset load: method 2
class RandomSamples(Dataset):
    def __init__(self, root_path, loader=nifti_numpy_loader, 
                 loader_dataset=make_dataset_subject, transform=None):
        super(RandomSamples, self).__init__()
        self.samples, self.targets = loader_dataset(root_path)



        self.loader = loader
        self.transform = transform     
        
    def __getitem__(self, index):
        """
        Args:
            root_path: simu_data/
            samples_folder: training/ or testing/
            targets_folder: label_training/ or label_testing/

        Returns:
            tuple: (sample, target) where target is simulation results.
        """
                
        sample = self.samples[index]
        target = self.targets[index]
        #print('sample dir:', sample)
        #print('target dir:', target)
        sample_loader = self.loader(sample)
        target_loader = self.loader(target)

        if self.transform is not None:
            sample_loader = self.transform(sample_loader)
            target_loader = self.transform(target_loader)

        return sample_loader, target_loader

    def __len__(self):
        return len(self.samples)  


# dataset load: load sample and gt
class LoadSampGT(Dataset):
    def __init__(self, root_path, loader=nifti_numpy_loader, 
                 loader_dataset=make_dataset_one_folder_cond, transform=None):
        super(LoadSampGT, self).__init__()
        self.samples, self.targets = loader_dataset(root_path)
        
        self.loader = loader
        self.transform = transform     
        
    def __getitem__(self, index):
        """
        Args:
            root_path: simu_data/
            samples_folder: training/ or testing/
            targets_folder: label_training/ or label_testing/

        Returns:
            tuple: (sample, target) where target is simulation results.
        """
                
        sample = self.samples[index]
        target = self.targets[index]
#        print('sample dir:', sample)
#        print('target dir:', target)
        sample_loader = self.loader(sample)
        target_loader = self.loader(target)

        if self.transform is not None:
            sample_loader = self.transform(sample_loader)
            target_loader = self.transform(target_loader)

        return sample_loader, target_loader

    def __len__(self):
        return len(self.samples)

# dataset load: load sample and gt
class LoadSampGT_T1(Dataset):
    def __init__(self, root_path_sample, root_path_label, loader=nifti_numpy_loader, 
                 loader_dataset=make_dataset_one_folder_T1D_v2, transform=None):
        super(LoadSampGT_T1, self).__init__()
        self.samples, self.targets = loader_dataset(root_path_sample, root_path_label)
        
        self.loader = loader
        self.transform = transform     
        
    def __getitem__(self, index):
        """
        Args:
            root_path: simu_data/
            samples_folder: training/ or testing/
            targets_folder: label_training/ or label_testing/

        Returns:
            tuple: (sample, target) where target is simulation results.
        """
                
        sample = self.samples[index]
        target = self.targets[index]
#        print('sample dir:', sample)
#        print('target dir:', target)
        sample_loader = self.loader(sample)
        target_loader = self.loader(target)

        if self.transform is not None:
            sample_loader = self.transform(sample_loader)
            target_loader = self.transform(target_loader)

        return sample_loader, target_loader

    def __len__(self):
        return len(self.samples)             

# dataset load: method 2
class RandomSamplesMask(Dataset):
    def __init__(self, root_path, loader=nifti_numpy_loader, transform=None):
        super(RandomSamplesMask, self).__init__()
        self.samples, self.targets, self.masks = make_dataset_mask(root_path)
        
        self.loader = loader
        self.transform = transform     
        
    def __getitem__(self, index):
        """
        Args:
            root_path: simu_data/
            samples_folder: training/ or testing/
            targets_folder: label_training/ or label_testing/

        Returns:
            tuple: (sample, target) where target is simulation results.
        """
        sample = self.samples[index]
        target = self.targets[index]
        mask = self.masks[index]
        sample_loader = self.loader(sample)
        target_loader = self.loader(target)
        mask_loader = self.loader(mask)
        if self.transform is not None:
            sample_loader = self.transform(sample_loader)
            target_loader = self.transform(target_loader)
            mask_loader = self.transform(mask_loader)

        return sample_loader, target_loader, mask_loader

    def __len__(self):
        return len(self.samples) 
    
# load testing samples without labels    
class LoadInputSample(Dataset):
    def __init__(self, samp_dirs, loader=nifti_numpy_loader, transform=None):
        super(LoadInputSample, self).__init__()
        self.samples = samp_dirs
        
        self.loader = loader
        self.transform = transform     
        
    def __getitem__(self, index):
        """
        Args:
            root_path: simu_data/
            samples_folder: training/ or testing/
            targets_folder: label_training/ or label_testing/

        Returns:
            tuple: (sample, target) where target is simulation results.
        """
                
        sample = self.samples[index]
#        print('sample dir:', sample)
        sample_loader = self.loader(sample)

        if self.transform is not None:
            sample_loader = self.transform(sample_loader)

        return sample_loader

    def __len__(self):
        return len(self.samples) 
