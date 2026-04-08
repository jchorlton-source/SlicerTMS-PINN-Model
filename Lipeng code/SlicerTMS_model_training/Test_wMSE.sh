#!/bin/bash

source /rfanfs/pnl-zorro/home/lipeng/bashrc3
conda activate tms
cd /NAS/Synology/MultiCoilData/BrainCSF/Iso/2mm/Code/GP_data/code/Network_train_toy

python test_R01_cpu.py Iso_wMSE_noinitial/Trained_wMSE_epoch_150.pth.tar Coil5 DNN_wMSE/Coil5/
python test_R01_cpu.py Iso_threecoils_noinitial/MultiCoil_MSE_epoch_900.pth.tar Coil5 DNN_allCoils/Coil5/
