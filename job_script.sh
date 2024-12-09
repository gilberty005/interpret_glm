#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --account pmg

python sae.py 