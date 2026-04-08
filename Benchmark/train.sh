#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=pawsey1242-gpu

export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1



module load pytorch/2.7.1-rocm6.3.3

export OMP_NUM_THREADS=1
export mycommand=$MYSCRATCH/Thesis/SlicerTMS-PINN-Model/benchmark/BenchmarkModel/newTrain.py
	
srun -N 1 -n 2 -c 16 --gres=gpu:4 --gpu-bind=closest python ${mycommand}

