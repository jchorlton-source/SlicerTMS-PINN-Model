#!/bin/bash
#SBATCH --job-name=ddp_train
#SBATCH --account=ems020
#SBATCH --partition=medical
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_out.log
#SBATCH --error=logs/%j_err.log

# Make sure log dir exists (some clusters fail silently if not)
mkdir -p logs

# --- DDP rendezvous ---
# srun launches one process per task (= one per GPU). newTrain.py reads
# SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS to populate RANK / LOCAL_RANK
# / WORLD_SIZE itself, so we just need MASTER_ADDR + MASTER_PORT here.
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# --- Threading hygiene ---
# Without this, each process spawns SLURM_CPUS_ON_NODE OpenMP threads
# and they fight each other. Pin to the per-task allocation.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- NCCL ---
# Fail loudly on async NCCL errors instead of hanging.
export NCCL_ASYNC_ERROR_HANDLING=1
# Uncomment if NCCL has issues and you need a verbose log:
# export NCCL_DEBUG=INFO

echo "Master:    $MASTER_ADDR"
echo "Nodes:     $SLURM_JOB_NODELIST"
echo "Tasks:     $SLURM_NTASKS"
echo "CPUs/task: $SLURM_CPUS_PER_TASK"

# srun launches one Python process per task; no torchrun needed for a
# single-node multi-GPU run. For multi-node, switch to the torchrun
# variant (--ntasks-per-node=1, torchrun spawns the workers).
srun ~/.local/bin/uv run python newTrain.py
