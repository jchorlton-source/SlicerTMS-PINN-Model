#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:23:31 2019

@author: jeg88
"""

import torch.nn as nn
import torch.nn.functional as F 
#from torch.nn import MSELoss   
from torch.autograd import Variable   
#class MSELoss(_Loss):
#    __constants__ = ['reduction']
#
#    def __init__(self, size_average=None, reduce=None, reduction='mean'):
#        super(MSELoss, self).__init__(size_average, reduce, reduction)
#
#    def forward(self, input, target):
#        return F.mse_loss(input, target, reduction=self.reduction)        
    
class MaskWeighedLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """
    def __init__(self, weight=None, reduction='mean'):
        super(MaskWeighedLoss, self).__init__()
        self.register_buffer('weight', weight)
#        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            input = input * weight
        return F.mse_loss(input, target, reduction='mean')

## another wey to define loss
def cal_mask_loss(output, target, mask):
    output = output * mask
    loss = F.mse_loss(output, target, reduction='mean')
    return loss        