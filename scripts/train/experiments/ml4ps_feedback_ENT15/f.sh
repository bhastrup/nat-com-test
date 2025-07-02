#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=sm3090el8
#SBATCH -N 1-1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --ntasks-per-node=8     # 8 MPI processes per node
#SBATCH --time=7-00:00:00 # num days-hour:min:sec of runtime (7 days max)
#SBATCH --gres=gpu:RTX3090:1 # Request 1 GPU (can increase for more)
#SBATCH --output=slurm_outputs/train/feedback_ent15/f_%j.out  # %j gives SLURM job ID
#SBATCH --error=slurm_outputs/train/feedback_ent15/f_%j.err   # %j gives SLURM job ID


# Pass the arguments from the bash script to the Python script
python scripts/train/experiments/ml4ps_feedback_ENT15/f.py