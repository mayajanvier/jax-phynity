#!/bin/bash
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err
#SBATCH --partition=electronic
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00


#export JAX_PLATFORMS=cpu
srun python inference.py --config config/config_ns_incomp_inf.yaml 