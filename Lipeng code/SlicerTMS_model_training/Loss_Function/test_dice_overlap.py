#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 09:36:59 2019

@author: gx020
"""
import numpy as np
import nibabel as nib
f_name = '/rfanfs/pnl-zorro/home/gp88/software/anaconda3/doc/guoping_test/proj1_doc/simu_data/testing/101107_TMS_0003_Ds.nii.gz'

# shape: (180, 220, 120, 6)
f_data = nib.load(f_name)
f_cond = f_data.get_data()[:,:,:,:3]
f_norm = np.power(f_cond, 2)
f_norm = np.sum(f_norm, axis=3)
f_norm = np.sqrt(f_norm)

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
f_brain_mask = (f_norm > 0)
#f_norm[f_brain_mask == 0]= np.NaN

# extract 5% region of brain tissue for simulation data
f_gt_name = '/rfanfs/pnl-zorro/home/gp88/software/anaconda3/doc/guoping_test/proj1_doc/simu_data/simu_target/101107_TMS_0003_E.nii.gz'
f_gt = nib.load(f_gt_name).get_data()
f_gt_norm = np.sqrt(np.sum(np.power(f_gt, 2), axis=3))
f_gt_norm_mask = f_gt_norm * f_brain_mask
label_gt = np.greater_equal(f_gt_norm_mask, np.percentile(f_gt_norm_mask[f_brain_mask], 95))

#label_gt = np.greater_equal(f_norm, np.percentile(f_norm[f_brain_mask], 90))


import matplotlib.pyplot as plt
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[i].axis('off')
#slice_0 = f_norm[:, :, 30]
#slice_1 = f_brain_mask[:, :, 30]
#slice_2 = label_gt[:, :, 30]
#show_slices([slice_0, slice_1, slice_2])
#plt.suptitle("Center slices for Cond image")  # doctest: +SKIP

# load predict data
import scipy.io as sio
f_pd_name = '/rfanfs/pnl-zorro/home/gp88/software/anaconda3/doc/guoping_test/proj1_doc/simu_data/simu_pred/101107_TMS_0003_E_pd.mat'
mat_contents = sio.loadmat(f_pd_name)
f_pd = mat_contents['pred']
f_pd = f_pd.transpose(2, 3, 4, 1, 0)
f_pd = np.squeeze(f_pd, axis=4)
f_pd_norm = np.sqrt(np.sum(np.power(f_pd, 2), axis=3))
f_pd_norm_mask = f_pd_norm * f_brain_mask
label_pd = np.greater_equal(f_pd_norm_mask, np.percentile(f_pd_norm_mask[f_brain_mask], 95))


slice_0 = f_norm[:, :, 30]
slice_1 = label_gt[:, :, 30]
slice_2 = label_pd[:, :, 30]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for Cond image")  # doctest: +SKIP


#def flatten(tensor):
#    """Flattens a given tensor such that the channel axis is first.
#    The shapes are transformed as follows:
#       (N, C, D, H, W) -> (C, N * D * H * W)
#    """
#    C = tensor.size(1)
#    # new axis order
#    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
#    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
#    transposed = tensor.permute(axis_order)
#    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
#    return transposed.view(C, -1)
# cal dice loss
label_gt = label_gt.flatten()
label_pd = label_pd.flatten()

# Compute per channel Dice Coefficient
intersect = (label_gt * label_pd).sum(-1)

denominator = (label_gt + label_pd).sum(-1) * 2
dice = (2. * intersect) / denominator.clip(0.0001, denominator.max())



