#!/bin/bash

#SBATCH -o slurm/run_convergence_test.log-%j
#SBATCH --partition=GPU-shared
#SBATCH --job-name=conv_test
##SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=24:00:00

export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export MKL_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5

module load anaconda3
source activate gegd_dev
module load cuda/12.6

##python /ocean/projects/cis260139p/smin2/gaussian_ensemble_gradient_descent/runfiles/RCWA_functions/polarization_beamsplitter_convergence_test_FMMAX.py
##python /ocean/projects/cis260139p/smin2/gaussian_ensemble_gradient_descent/runfiles/RCWA_functions/RGB_coupler_convergence_test_FMMAX.py
python /ocean/projects/cis260139p/smin2/gaussian_ensemble_gradient_descent/runfiles/RCWA_functions/RGB_color_router_convergence_test_FMMAX.py

##python /ocean/projects/cis260139p/smin2/gaussian_ensemble_gradient_descent/runfiles/FDTD_functions/WDM_diplexer_convergence_test_ceviche.py

##export QT_QPA_PLATFORM=offscreen
##python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/integrated_bandpass_filter_convergence_test_lumFDTD.py --Nthreads 12
##python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/mode_converter_convergence_test_lumFDTD.py --Nthreads 12
##python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/FDTD_functions/PRS_convergence_test_lumFDTD.py --Nthreads 48

##SBATCH --ntasks=32

##/opt/lumerical/v231/mpich2/nemesis/bin/mpiexec -n 32 python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/mode_converter_convergence_test_MEEP.py --Nthreads -1

##SBATCH --ntasks=1
##SBATCH --cpus-per-task=32

##python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/mode_converter_convergence_test_ceviche.py --Nthreads 32