#!/bin/bash
#SBATCH --time=0-05:00            
#SBATCH --mem=16G                
#SBATCH --cpus-per-task=12         
#SBATCH --gpus-per-node=24
#SBATCH --account pmg

python sae.py 