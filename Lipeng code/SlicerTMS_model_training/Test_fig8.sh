#!/bin/bash

source /rfanfs/pnl-zorro/home/lipeng/bashrc3
conda activate tms
cd /NAS/Synology/MultiCoilData/BrainCSF/Iso/2mm/Code/GP_data/code/Network_train_toy

python test_R01_cpu.py Iso_fig8/Trained_isocond_epoch_1000.pth.tar Coil1 DNN_fig8/Coil1/
python test_R01_cpu.py Iso_fig8/Trained_isocond_epoch_1000.pth.tar Coil2 DNN_fig8/Coil2/
python test_R01_cpu.py Iso_fig8/Trained_isocond_epoch_1000.pth.tar Coil5 DNN_fig8/Coil5/
python test_R01_cpu.py Iso_fig8/Trained_isocond_epoch_1000.pth.tar Coil6 DNN_fig8/Coil6/