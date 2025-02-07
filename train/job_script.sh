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

# Load Mamba
module load mamba

# Ensure Conda is initialized correctly
source /cm/shared/apps/mambaforge/etc/profile.d/conda.sh

# Define the environment name and path
ENV_NAME="py10"
ENV_PATH="/burg/pmg/users/gy2322/conda_envs/$ENV_NAME"
ENV_YAML="/burg/pmg/users/gy2322/environment.yaml"

# Check if the environment exists; if not, create it
if [ ! -d "$ENV_PATH" ]; then
    echo "Creating Conda environment from environment.yaml..."
    conda env create -f "$ENV_YAML"
else
    echo "Updating Conda environment..."
    conda env update --file "$ENV_YAML" --prune
fi

# Activate the environment
conda activate "$ENV_PATH"

# Debugging: Verify Python and PyTorch installation
echo "Using Python from: $(which python)"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Run the script
python sae_revised.py