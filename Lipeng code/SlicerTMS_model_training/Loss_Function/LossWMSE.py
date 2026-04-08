#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:23:31 2019

@author: jeg88
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 
#from torch.nn import MSELoss   
from torch.autograd import Variable   


class wMSELoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """
    def __init__(self, weight=None, reduction='mean'):
        super(wMSELoss, self).__init__()
        self.register_buffer('weight', weight)
#        self.ignore_index = ignore_index

    def forward(self, input, target):
        w = target**2
        w = .2*w.sum(dim=1)+1
        w = w.unsqueeze(1)
        w = torch.broadcast_to(w,target.size())
        return F.mse_loss(w*input, w*target, size_average=True, reduction='mean')
