#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:19:30 2019

@author: jeg88
"""

import os
import LoadDataMask as LD
root_path = '/rfanfs/pnl-zorro/home/gp88/tms_data'

samp, tar = LD.make_dataset_subject(root_path)