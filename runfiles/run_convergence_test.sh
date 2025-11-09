#!/bin/bash

#SBATCH -o /home/minseokhwan/Ensemble_Optimization/slurm/run_convergence_test.log-%j
#SBATCH --partition=48core
##SBATCH --nodelist=node3
##SBATCH --exclusive
#SBATCH --job-name=conv_test

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

##python /home/minseokhwan/Ensemble_Optimization/TORCWA_functions/diffraction_grating_convergence_test.py --Nthreads 32

export QT_QPA_PLATFORM=offscreen
##python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/integrated_bandpass_filter_convergence_test_lumFDTD.py --Nthreads 12
python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/mode_converter_convergence_test_lumFDTD.py --Nthreads 12

##SBATCH --ntasks=32

##/opt/lumerical/v231/mpich2/nemesis/bin/mpiexec -n 32 python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/mode_converter_convergence_test_MEEP.py --Nthreads -1

##SBATCH --ntasks=1
##SBATCH --cpus-per-task=32

##python /home/minseokhwan/Ensemble_Optimization/FDTD_functions/mode_converter_convergence_test_ceviche.py --Nthreads 32