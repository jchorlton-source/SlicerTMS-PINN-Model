#!/bin/bash
#SBATCH --account=ems020
#SBATCH --partition=medical
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/sanity_%j.log

export PATH="$HOME/.local/bin:/group/ems020/jchorlton/.local/bin:$PATH"

/group/ems020/jchorlton/.local/bin/uv run python runInference.py
