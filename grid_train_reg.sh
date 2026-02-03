#!/bin/bash
#SBATCH --job-name=base_ks
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err
#SBATCH --time=2-00:00:00

export JAX_PLATFORM_NAME=cpu
# run baseline with only trajectory loss
srun python train_jaxphynity.py --config config/config_ks.yaml

# launch experiments for different hyperparameters
# lambdas = ( 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 2.0 10.0 )
# # loss lip
# for lambda in ${lambdas[@]}
# do
#    srun python train_jaxphynity.py --config configs/lorenz.yaml train.lambda0=$lambda  train.reg_loss_name=loss_Fa_prime_supervisedX model.phy_option=none_Fa_prime_supX
# done

# # loss Fa
# for lambda in ${lambdas[@]}
# do
#    srun python train_jaxphynity.py --config configs/lorenz.yaml train.lambda0=$lambda  train.reg_loss_name=loss_Fa model.phy_option=none_Fa
# done