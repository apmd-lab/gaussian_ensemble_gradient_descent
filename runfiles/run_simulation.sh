#!/bin/bash

#SBATCH -o slurm/run_simulation.log-%j
#SBATCH --partition=GPU-shared
#SBATCH --job-name=ens_sim
##SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=1:00:00

export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export MKL_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5

module load anaconda3
source activate gegd_dev
module load cuda/12.6

##python /ocean/projects/cis260139p/smin2/gaussian_ensemble_gradient_descent/runfiles/simulate_optimized_polarization_beamsplitter.py
python /ocean/projects/cis260139p/smin2/gaussian_ensemble_gradient_descent/runfiles/simulate_optimized_RGB_coupler.py
##python /ocean/projects/cis260139p/smin2/gaussian_ensemble_gradient_descent/runfiles/simulate_optimized_RGB_color_router.py