#!/bin/bash
#SBATCH --job-name=train_sae          # Job name
#SBATCH --output=train_sae.out        # Output file
#SBATCH --error=train_sae.err         # Error file
#SBATCH --time=30:00:00               # Max runtime (format: HH:MM:SS)
#SBATCH --partition=pmg               # Partition to submit to
#SBATCH --gres=gpu:8                  # Request 1 GPU (adjust as needed)
#SBATCH --cpus-per-task=8             # Request 8 CPUs (adjust as needed)
#SBATCH --mem=32G                     # Request 32GB of memory (adjust as needed)
#SBATCH --account=pmg                 # Specify account (if required)

module load mamba

source activate /burg/pmg/users/gy2322/conda_envs/py10

python sae_revised.py