#!/bin/bash

#SBATCH -o /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/slurm/run_convergence_test.log-%j
#SBATCH --partition=48core
#SBATCH --nodelist=node2
#SBATCH --exclusive
#SBATCH --job-name=conv_test

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

##python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/RCWA_functions/polarization_beamsplitter_convergence_test_FMMAX.py
##python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/RCWA_functions/RGB_coupler_convergence_test_FMMAX.py
python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/RCWA_functions/RGB_color_router_convergence_test_FMMAX.py

##python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/FDTD_functions/WDM_diplexer_convergence_test_ceviche.py

##export QT_QPA_PLATFORM=offscreen
##python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/integrated_bandpass_filter_convergence_test_lumFDTD.py --Nthreads 12
##python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/mode_converter_convergence_test_lumFDTD.py --Nthreads 12
##python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/FDTD_functions/PRS_convergence_test_lumFDTD.py --Nthreads 48

##SBATCH --ntasks=32

##/opt/lumerical/v231/mpich2/nemesis/bin/mpiexec -n 32 python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/mode_converter_convergence_test_MEEP.py --Nthreads -1

##SBATCH --ntasks=1
##SBATCH --cpus-per-task=32

##python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/mode_converter_convergence_test_ceviche.py --Nthreads 32