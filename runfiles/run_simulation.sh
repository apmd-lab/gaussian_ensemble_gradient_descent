#!/bin/bash

#SBATCH -o slurm/run_simulation.log-%j
#SBATCH --partition=ghx4
#SBATCH --job-name=ens_sim
##SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
##SBATCH --gres=gpu:h100pcie:1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --account=bhkk-dtai-gh

export OMP_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export MKL_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20

module load python/miniforge3_pytorch/2.11.0
conda activate base
source /work/nvme/bhkk/smin2/myenv/bin/activate

##export XLA_PYTHON_CLIENT_PREALLOCATE=false

##python simulate_optimized_polarization_beamsplitter.py
##python simulate_optimized_RGB_coupler.py
python simulate_optimized_RGB_color_router.py