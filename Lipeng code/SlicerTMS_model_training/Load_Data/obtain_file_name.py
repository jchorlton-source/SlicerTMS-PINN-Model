#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:35:49 2019

@author: gx020
"""

import os
import fnmatch
# 116524/
root_path = '/run/media/gx020/TMSData/error_subject/116524'

#file_name = [x for x in os.listdir(root_path)]

# write it to a file
write_name_file = open('/run/media/gx020/TMSData/error_subject/sample_name_t1d_116524.txt', 'a')

for file in os.listdir(root_path):
    if fnmatch.fnmatch(file, '*T1D.nii.gz'):
#        print(file)
        file_name = os.path.splitext(file)[0]
        file_name = os.path.splitext(file_name)[0]
        write_name_file.write(file_name + '\n')

write_name_file.close()
        
