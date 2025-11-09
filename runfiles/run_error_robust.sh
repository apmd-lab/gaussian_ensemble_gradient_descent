#!/bin/bash

#SBATCH -o /home/minseokhwan/Ensemble_Optimization/slurm/run_error_robust.log-%j
#SBATCH --partition=32core
#SBATCH --nodelist=node3
#SBATCH --job-name=err_robust
##SBATCH --exclusive

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

## Diffraction Grating ---------------------------------------------------
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

python /home/minseokhwan/Ensemble_Optimization/run_error_robustness_diffraction_grating.py --Nthreads 20 --upsample_ratio 1

## Integrated Bandpass Filter ---------------------------------------------------
##export QT_QPA_PLATFORM=offscreen

##SBATCH --ntasks=1
##SBATCH --cpus-per-task=12

##python /home/minseokhwan/Ensemble_Optimization/run_error_robustness_mode_converter.py --Nthreads 12 --upsample_ratio 1