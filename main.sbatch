#!/bin/bash

#SBATCH -q ebrainccs
#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=47:59:59
#SBATCH --mem=100GB
#SBATCH --job-name=jailmine

module purge

# Load the Conda module
source ~/.bashrc

# Activate your Conda environment
conda activate jailmine

export TRANSFORMERS_CACHE="/scratch/bc3194/huggingface_cache"
export HF_HOME="/scratch/bc3194/huggingface_cache"
python -u main.py