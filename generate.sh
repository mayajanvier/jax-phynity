#!/bin/bash
#SBATCH --job-name=gen_KS
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.err
#SBATCH --time=24:00:00


# train
#python LPSDA/generate/generate_data.py --experiment=KS --train_samples=512 --valid_samples=0 --test_samples=0 --L=64 --nt=500

# val 
#python LPSDA/generate/generate_data.py --experiment=KS --train_samples=0 --valid_samples=128 --test_samples=0 --L=64 --nt=500

# val long 
python LPSDA/generate/generate_data.py --experiment=KS --train_samples=0 --valid_samples=128 --test_samples=0 --L=64 --nt=1000 --nt_effective=640 --end_time=200

# test
#python LPSDA/generate/generate_data.py --experiment=KS --train_samples=0 --valid_samples=0 --test_samples=128 --L=64 --nt=1000 --nt_effective=640 --end_time=200