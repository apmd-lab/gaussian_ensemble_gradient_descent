#!/bin/bash

#SBATCH -o slurm/run_optimization.log-%j
#SBATCH --job-name=gegd9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
##SBATCH --gres=gpu:h100-80:1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00

export OMP_NUM_THREADS=72
export OPENBLAS_NUM_THREADS=72
export MKL_NUM_THREADS=72
export NUMEXPR_NUM_THREADS=72

##source "$(dirname "$0")/env_seneca.sh"
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/env_deltaAI.sh"
##source "$(dirname "$0")/env_anvilAI.sh"

## Polarization Beamsplitter -----------------------------------------------------
##: << 'END_COMMENT'
python run_optimization_polarization_beamsplitter.py \
    --Nthreads 72 \
    --n_seed 9 \
    --load_data 0 \
    --optimizer 'GEGD' \
    --Nensemble 20 \
    --Nx 45 \
    --Ny 90 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --coeff_exp 30 \
    --maxiter -1 \
    --sigma_ensemble 1e-2 \
    --eta 5e-5 \
    --min_feature_size 7 \
    --precision 'float64'
##END_COMMENT

## RGB Coupler -----------------------------------------------------
: << 'END_COMMENT'
python run_optimization_RGB_coupler.py \
    --Nthreads 72 \
    --n_seed 7 \
    --load_data 0 \
    --optimizer 'GEGD' \
    --Nensemble 20 \
    --Nx 60 \
    --Ny 263 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --coeff_exp 20 \
    --maxiter -1 \
    --sigma_ensemble 1e-2 \
    --eta 5e-5 \
    --min_feature_size 7 \
    --precision 'float64'
END_COMMENT

## RGB Color Router -----------------------------------------------------
: << 'END_COMMENT'
python run_optimization_RGB_color_router.py \
    --Nthreads 72 \
    --n_seed 9 \
    --load_data 0 \
    --optimizer 'GEGD' \
    --Nensemble 20 \
    --Nx 100 \
    --Ny 100 \
    --symmetry 3 \
    --upsample_ratio 1 \
    --coeff_exp 20 \
    --maxiter -1 \
    --sigma_ensemble 1e-2 \
    --eta 5e-5 \
    --min_feature_size 7 \
    --precision 'float64'
END_COMMENT