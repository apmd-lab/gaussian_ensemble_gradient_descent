#!/bin/bash

#SBATCH -o slurm/run_optimization.log-%j
#SBATCH --partition=cac_gpu
#SBATCH --job-name=ens_opt
##SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export MKL_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20

module load anaconda3
unset PYTHONPATH PYTHONHOME
source activate gegd_dev
module load cuda/12.9

## Polarization Beamsplitter -----------------------------------------------------
: << 'END_COMMENT'
python run_optimization_polarization_beamsplitter.py \
    --Nthreads 20 \
    --n_seed 9 \
    --load_data 0 \
    --optimizer 'sep_CMA_ES' \
    --Nensemble 20 \
    --Nx 45 \
    --Ny 90 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --maxiter 400 \
    --min_feature_size 7 \
    --precision 'float64'
END_COMMENT

## RGB Coupler -----------------------------------------------------
: << 'END_COMMENT'
python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_RGB_coupler.py \
    --Nthreads 8 \
    --n_seed 0 \
    --load_data 0 \
    --optimizer 'sep_CMA_ES' \
    --Nensemble 20 \
    --Nx 60 \
    --Ny 263 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --maxiter 300 \
    --min_feature_size 7 \
    --precision 'float64'
END_COMMENT

## RGB Color Router -----------------------------------------------------
##: << 'END_COMMENT'
python run_optimization_RGB_color_router.py \
    --Nthreads 20 \
    --n_seed 8 \
    --load_data 0 \
    --optimizer 'sep_CMA_ES' \
    --Nensemble 20 \
    --Nx 100 \
    --Ny 100 \
    --symmetry 3 \
    --upsample_ratio 1 \
    --maxiter 400 \
    --min_feature_size 7 \
    --precision 'float64'
##END_COMMENT