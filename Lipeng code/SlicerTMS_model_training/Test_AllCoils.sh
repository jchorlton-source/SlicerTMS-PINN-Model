#!/bin/bash

#source /rfanfs/pnl-zorro/home/lipeng/bashrc3
#conda activate tms
cd /NAS/Synology/MultiCoilData/BrainCSF/Iso/2mm/Code/GP_data/code/Network_train_toy

python test_R01_cpu.py Iso_threecoils_noinitial/MultiCoil_MSE_epoch_900.pth.tar Coil5 DNN_allCoils/Coil5/
python test_R01_cpu.py Iso_threecoils_noinitial/MultiCoil_MSE_epoch_900.pth.tar Coil13 DNN_allCoils/Coil13/
python test_R01_cpu.py Iso_threecoils_noinitial/MultiCoil_MSE_epoch_900.pth.tar Coil18 DNN_allCoils/Coil18/