#!/bin/bash
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err


export JAX_PLATFORMS=cpu
srun python inference.py --config config/config_twobody_inf.yaml 