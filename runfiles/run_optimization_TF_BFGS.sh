#!/bin/bash

#SBATCH -o slurm/run_optimization.log-%j
#SBATCH --partition=GPU-shared
#SBATCH --job-name=ens_opt
##SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=48:00:00

export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export MKL_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5

module load anaconda3
source activate gegd_dev
module load cuda/12.6

## Test Function RBF -----------------------------------------------------

##python run_optimization_test_functions.py \
##    --Nthreads 10 \
##    --n_seed 0 \
##    --load_data 0 \
##    --optimizer 'TF_BFGS' \
##    --Nensemble 10 \
##    --Nx 37 \
##    --Ny 269 \
##    --symmetry 1 \
##    --upsample_ratio 1 \
##    --maxiter 500 \
##    --min_feature_size 9

## Polarization Beamsplitter -----------------------------------------------------
: << 'END_COMMENT'
python run_optimization_polarization_beamsplitter.py \
    --Nthreads 8 \
    --n_seed 0 \
    --load_data 0 \
    --optimizer 'TF_BFGS' \
    --Nensemble 10 \
    --Nx 45 \
    --Ny 90 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --maxiter 300 \
    --min_feature_size 7
END_COMMENT

## RGB Coupler -----------------------------------------------------
: << 'END_COMMENT'
python run_optimization_RGB_coupler.py \
    --Nthreads 8 \
    --n_seed 3 \
    --load_data 0 \
    --optimizer 'TF_BFGS' \
    --Nensemble 20 \
    --Nx 60 \
    --Ny 263 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --maxiter 400 \
    --min_feature_size 7
END_COMMENT

## RGB Color Router -----------------------------------------------------
##: << 'END_COMMENT'
python run_optimization_RGB_color_router.py \
    --Nthreads 5 \
    --n_seed 0 \
    --load_data 0 \
    --optimizer 'TF_BFGS' \
    --Nensemble 20 \
    --Nx 100 \
    --Ny 100 \
    --symmetry 3 \
    --upsample_ratio 1 \
    --maxiter 500 \
    --min_feature_size 7
##END_COMMENT