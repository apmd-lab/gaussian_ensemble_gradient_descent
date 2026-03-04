#!/bin/bash

#SBATCH -o /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/slurm/run_optimization.log-%j
#SBATCH --partition=48core
#SBATCH --nodelist=node2
#SBATCH --job-name=ens_opt
##SBATCH --exclusive

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

## Test Function RBF -----------------------------------------------------

##python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_test_functions.py \
##    --Nthreads 48 \
##    --n_seed 0 \
##    --load_data 0 \
##    --optimizer 'GEGD' \
##    --Nensemble 10 \
##    --Nx 37 \
##    --Ny 269 \
##    --symmetry 1 \
##    --upsample_ratio 1 \
##    --coeff_exp 20 \
##    --maxiter 500 \
##    --sigma_ensemble 1e-2 \
##    --eta 1e-5 \
##    --min_feature_size 9

## Polarization Beamsplitter -----------------------------------------------------
: << 'END_COMMENT'
python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_polarization_beamsplitter.py \
    --Nthreads 8 \
    --n_seed 0 \
    --load_data 0 \
    --optimizer 'GEGD' \
    --Nensemble 20 \
    --Nx 45 \
    --Ny 90 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --coeff_exp 20 \
    --maxiter 300 \
    --sigma_ensemble 1e-2 \
    --eta 1e-4 \
    --min_feature_size 7
END_COMMENT

## RGB Coupler -----------------------------------------------------
: << 'END_COMMENT'
python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_RGB_coupler.py \
    --Nthreads 8 \
    --n_seed 9 \
    --load_data 0 \
    --optimizer 'GEGD' \
    --Nensemble 20 \
    --Nx 60 \
    --Ny 263 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --coeff_exp 20 \
    --maxiter 400 \
    --sigma_ensemble 1e-2 \
    --eta 5e-5 \
    --min_feature_size 7
END_COMMENT

## RGB Color Router -----------------------------------------------------
##: << 'END_COMMENT'
python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_RGB_color_router.py \
    --Nthreads 16 \
    --n_seed 0 \
    --load_data 0 \
    --optimizer 'GEGD' \
    --Nensemble 20 \
    --Nx 100 \
    --Ny 100 \
    --symmetry 3 \
    --upsample_ratio 1 \
    --coeff_exp 20 \
    --maxiter 300 \
    --sigma_ensemble 1e-2 \
    --eta 5e-5 \
    --min_feature_size 7
##END_COMMENT