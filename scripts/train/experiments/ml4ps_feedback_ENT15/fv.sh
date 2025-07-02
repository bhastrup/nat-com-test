#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=sm3090el8 # a100 # sm3090_devel # sm3090el8 a100_week
#SBATCH -N 1-1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --ntasks-per-node=8     # 8 MPI processes per node
#SBATCH --time=2-00:00:00 # num days-hour:min:sec of runtime (7 days max)
#SBATCH --gres=gpu:1 # Request 1 GPU (can increase for more) # gpu:RTX3090:1 
#SBATCH --output=slurm_outputs/train/feedback_ent15/fv_%j.out  # %j gives SLURM job ID
#SBATCH --error=slurm_outputs/train/feedback_ent15/fv_%j.err   # %j gives SLURM job ID


# Pass the arguments from the bash script to the Python script
python scripts/train/experiments/ml4ps_feedback_ENT15/fv.py