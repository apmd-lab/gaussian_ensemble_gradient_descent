#!/bin/bash

#SBATCH -o slurm/run_simulation.log-%j
#SBATCH --job-name=ens_sim
##SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
##SBATCH --gres=gpu:h100pcie:1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00

export OMP_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export MKL_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20

##source "$(dirname "$0")/env_seneca.sh"
##source "$(dirname "$0")/env_deltaAI.sh"
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/env_anvilAI.sh"

python simulate_optimized_polarization_beamsplitter.py
##python simulate_optimized_RGB_coupler.py
##python simulate_optimized_RGB_color_router.py