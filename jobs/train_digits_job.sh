#!/bin/bash
#SBATCH --job-name=research_task
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/data/slurm_logs/slurm_job%j_research_task.out
#SBATCH --error=/data/slurm_logs/slurm_job%j_research_task.err

# Load environment
module load cuda/12.8
module load cudnn/9.9
module load mamba
micromamba activate columns_env

# Prevent VRAM pre-allocation errors
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run your code
python /home/dasja/projects/ImageColumnNetwork/scripts/test_cluster.py
