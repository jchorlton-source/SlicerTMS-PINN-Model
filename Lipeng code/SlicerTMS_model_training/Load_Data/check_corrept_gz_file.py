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

# check the nii.gz whether corrept
import nibabel as nib
import LoadData as LD
import os
import matplotlib.pyplot as plt

root_path = '/run/media/gx020/TMSData'
train_path = os.path.join(root_path, 'training_data')
sample_dir_all, target_dir_all = LD.make_dataset_subject_T1(train_path)
        
for f_name1, f_name2 in zip(sample_dir_all,target_dir_all):
    print(f_name1)
    nib.load(f_name1).get_data() 
    print(f_name2)
    nib.load(f_name2).get_data()     
    
    
  