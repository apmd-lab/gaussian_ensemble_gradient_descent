#!/bin/bash

#SBATCH -o slurm/run_ensemble_gradient_cosine_similarity.log-%j
#SBATCH --partition=ghx4
#SBATCH --job-name=cos_sim
##SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
##SBATCH --gres=gpu:h100-80:1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --account=bhkk-dtai-gh

export OMP_NUM_THREADS=72
export OPENBLAS_NUM_THREADS=72
export MKL_NUM_THREADS=72
export NUMEXPR_NUM_THREADS=72

module load python/miniforge3_pytorch/2.11.0
conda activate base
source /work/nvme/bhkk/smin2/myenv/bin/activate

## Polarization Beamsplitter -----------------------------------------------------
python run_ensemble_gradient_cosine_similarity.py \
    --Nthreads 72 \
    --n_seed 1 \
    --load_data 1 \
    --optimizer 'GEGD' \
    --Nensemble 20 \
    --Nx 45 \
    --Ny 90 \
    --symmetry 1 \
    --upsample_ratio 1 \
    --coeff_exp 20 \
    --maxiter -1 \
    --sigma_ensemble 1e-2 \
    --eta 5e-5 \
    --min_feature_size 7 \
    --precision 'float64'