#!/bin/bash

#SBATCH -o slurm/run_simulation.log-%j
#SBATCH --partition=cac_gpu
#SBATCH --job-name=ens_sim
##SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:h100pcie:1
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export MKL_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20

module load anaconda3
unset PYTHONPATH PYTHONHOME
source activate gegd_dev
module load cuda/12.9

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python simulate_optimized_polarization_beamsplitter.py
##python simulate_optimized_RGB_coupler.py
##python simulate_optimized_RGB_color_router.py