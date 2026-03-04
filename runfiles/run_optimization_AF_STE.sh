#!/bin/bash

#SBATCH -o /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/slurm/run_optimization.log-%j
#SBATCH --partition=48core
#SBATCH --nodelist=node1
#SBATCH --job-name=ens_opt
##SBATCH --exclusive

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

## Test Function RBF -----------------------------------------------------

##python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_test_functions.py \
##    --Nthreads 10 \
##    --n_seed 0 \
##    --load_data 0 \
##    --optimizer 'sep_CMA_ES' \
##    --Nensemble 10 \
##    --Nx 37 \
##    --Ny 269 \
##    --symmetry 1 \
##    --upsample_ratio 1 \
##    --maxiter 500 \
##    --min_feature_size 9

## Polarization Beamsplitter -----------------------------------------------------
##: << 'END_COMMENT'
python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_polarization_beamsplitter.py \
    --Nthreads 8 \
    --n_seed 0 \
    --load_data 0 \
    --optimizer 'AF_STE' \
    --Nensemble 20 \
    --Nx 45 \
    --Ny 90 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --maxiter 300 \
    --min_feature_size 7 \
    --eta 0.01
##END_COMMENT

## RGB Coupler -----------------------------------------------------
: << 'END_COMMENT'
python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_RGB_coupler.py \
    --Nthreads 8 \
    --n_seed 0 \
    --load_data 0 \
    --optimizer 'AF_STE' \
    --Nensemble 20 \
    --Nx 60 \
    --Ny 263 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --maxiter 300 \
    --min_feature_size 7 \
    --eta 0.01
END_COMMENT