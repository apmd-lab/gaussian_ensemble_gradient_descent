#!/bin/bash

#SBATCH -o /home/minseokhwan/Ensemble_Optimization/slurm/run_simulation.log-%j
#SBATCH --partition=32core
#SBATCH --nodelist=node4
#SBATCH --job-name=ens_sim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

## Diffraction Grating

##python /home/minseokhwan/Ensemble_Optimization/simulate_optimized_diffraction_grating.py --Nthreads 32 --upsample_ratio 1

export QT_QPA_PLATFORM=offscreen

python /home/minseokhwan/Ensemble_Optimization/simulate_optimized_mode_converter.py --Nthreads 12 --brush_size 7 --upsample_ratio 1