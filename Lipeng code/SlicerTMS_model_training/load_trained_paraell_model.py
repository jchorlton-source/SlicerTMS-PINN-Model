#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 18:22:25 2019

@author: gx020
"""
import torch
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in torch.load('model_subj20_cond_res_iter_1.pth').items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v