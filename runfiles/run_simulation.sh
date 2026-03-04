#!/bin/bash

#SBATCH -o /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/slurm/run_simulation.log-%j
#SBATCH --partition=32core
#SBATCH --nodelist=node3
#SBATCH --job-name=ens_sim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

##python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/simulate_optimized_polarization_beamsplitter.py
python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/simulate_optimized_RGB_coupler.py
##python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/simulate_optimized_RGB_color_router.py