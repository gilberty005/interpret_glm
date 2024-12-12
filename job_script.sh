#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=8
#SBATCH --account pmg

source activate pytorch_env

python sae.py 