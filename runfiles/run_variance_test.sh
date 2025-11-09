#!/bin/bash

#SBATCH -o /home/minseokhwan/Ensemble_Optimization/slurm/run_variance_test.log-%j
#SBATCH --partition=32core
##SBATCH --nodelist=node4
#SBATCH --job-name=var_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python /home/minseokhwan/Ensemble_Optimization/run_variance_test_test_functions.py --Nthreads 32 --upsample_ratio 1 --brush_size 7