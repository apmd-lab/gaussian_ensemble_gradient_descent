#!/bin/bash

#SBATCH -o slurm/run_optimization.log-%j
#SBATCH --partition=ghx4
#SBATCH --job-name=ga1
##SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
##SBATCH --gres=gpu:h100:1
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --account=bhkk-dtai-gh

export OMP_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export MKL_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20

module load python/miniforge3_pytorch/2.11.0
conda activate base
source /work/nvme/bhkk/smin2/myenv/bin/activate

## Polarization Beamsplitter -----------------------------------------------------
##: << 'END_COMMENT'
python run_optimization_polarization_beamsplitter.py \
    --Nthreads 20 \
    --n_seed 1 \
    --load_data 0 \
    --optimizer 'GA' \
    --Nensemble 20 \
    --Nx 45 \
    --Ny 90 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --maxiter -1 \
    --min_feature_size 7 \
    --precision 'float64'
##END_COMMENT
## load_data = 1 for n_seed = 0, 1, 2, 3, 5

## RGB Coupler -----------------------------------------------------
: << 'END_COMMENT'
python run_optimization_RGB_coupler.py \
    --Nthreads 20 \
    --n_seed 5 \
    --load_data 1 \
    --optimizer 'AF_GA' \
    --Nensemble 20 \
    --Nx 60 \
    --Ny 263 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --maxiter 400 \
    --min_feature_size 7 \
    --precision 'float64'
END_COMMENT

## RGB Color Router -----------------------------------------------------
: << 'END_COMMENT'
python run_optimization_RGB_color_router.py \
    --Nthreads 20 \
    --n_seed 9 \
    --load_data 0 \
    --optimizer 'AF_GA' \
    --Nensemble 20 \
    --Nx 100 \
    --Ny 100 \
    --symmetry 3 \
    --upsample_ratio 1 \
    --maxiter 400 \
    --min_feature_size 7 \
    --precision 'float64'
END_COMMENT
## load_data = 1 for n_seed = 8