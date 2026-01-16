#!/bin/bash

#SBATCH -o /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/slurm/run_optimization.log-%j
#SBATCH --partition=48core
#SBATCH --nodelist=node1
#SBATCH --job-name=ens_opt
##SBATCH --exclusive

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

## Test Function RBF -----------------------------------------------------
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

## TF_BFGS
##for i in {0..9}
##do
##    python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_test_functions.py --Nthreads 12 --n_seed $i --load_data 0 --optimizer 'TF_BFGS' --symmetry 1 --upsample_ratio 1 --maxiter 500 --brush_size 7
##done

## AF_STE
##for i in {0..9}
##do
##    python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_test_functions.py --Nthreads 12 --n_seed $i --load_data 0 --optimizer 'AF_STE' --symmetry 1 --upsample_ratio 1 --maxiter 500 --eta 0.001 --brush_size 7
##done

## ACMA_ES
for i in {0..9}
do
    python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_test_functions.py --Nthreads 12 --n_seed $i --load_data 0 --optimizer 'ACMA_ES' --Nx 105 --Ny 105 --symmetry 0 --upsample_ratio 1 --maxiter 200 --eta 0.1 --min_feature_size 7 --zoom_factor 1.0
done

## AF_PSO
##for i in {0..9}
##do
##    python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_test_functions.py --Nthreads 12 --n_seed $i --load_data 0 --optimizer 'AF_PSO' --symmetry 1 --upsample_ratio 1 --maxiter 500 --brush_size 7
##done

## AF_GA
##for i in {0..9}
##do
##    python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_test_functions.py --Nthreads 12 --n_seed $i --load_data 0 --optimizer 'AF_GA' --symmetry 1 --upsample_ratio 1 --maxiter 500 --brush_size 7
##done

## AF_NES
##for i in {0..9}
##do
##    python /home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles/run_optimization_test_functions.py --Nthreads 12 --n_seed $i --load_data 0 --optimizer 'AF_NES' --symmetry 4 --upsample_ratio 1 --maxiter 500 --sigma_ensemble_max 0.001 --eta 0.000001 --brush_size 7
##done

## Diffraction Grating ---------------------------------------------------
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=48

## Grayscale
##for i in {0..9}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_diffraction_grating.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'grayscale' --upsample_ratio 1 --maxiter 500 --binary 0 --eta 0.01 --brush_size 7
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_diffraction_grating.py --Nthreads 32 --n_seed $i --load_data 1 --optimizer 'grayscale' --upsample_ratio 1 --maxiter 500 --binary 0 --eta 0.01 --brush_size 7
##done

##python /home/minseokhwan/Ensemble_Optimization/run_optimization_diffraction_grating.py --Nthreads 48 --n_seed 9 --load_data 0 --optimizer 'grayscale' --upsample_ratio 1 --maxiter 500 --brush_size 7
##python /home/minseokhwan/Ensemble_Optimization/run_optimization_diffraction_grating.py --Nthreads 48 --n_seed 9 --load_data 1 --optimizer 'grayscale' --upsample_ratio 1 --maxiter 500 --brush_size 7

## Brush
##for i in {0..9}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_diffraction_grating.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'conventional' --upsample_ratio 1 --maxiter 500 --binary 1 --eta 0.001 --brush_size 7
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_diffraction_grating.py --Nthreads 32 --n_seed $i --load_data 1 --optimizer 'conventional' --upsample_ratio 1 --maxiter 500 --binary 1 --eta 0.001 --brush_size 7
##done

## Ensemble
##for i in {0..9}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_diffraction_grating.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'ensemble' --upsample_ratio 1 --coeff_exp 20 --maxiter 500 --sigma_ensemble_max 0.01 --eta 0.0001 --brush_size 7
##done

##for i in 4 5 8 9;
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_diffraction_grating.py --Nthreads 48 --n_seed $i --load_data 0 --optimizer 'ensemble' --upsample_ratio 3 --coeff_exp 20 --maxiter 500 --sigma_ensemble_max 0.01 --eta 0.0001 --brush_size 7
##done

##python /home/minseokhwan/Ensemble_Optimization/run_optimization_diffraction_grating.py --Nthreads 48 --n_seed 1 --load_data 0 --optimizer 'ensemble' --upsample_ratio 3 --coeff_exp 20 --maxiter 500 --sigma_ensemble_max 0.01 --eta 0.0001 --brush_size 7

## PSO
##for i in {3..9}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_diffraction_grating.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'PSO' --upsample_ratio 1 --maxiter 500 --brush_size 7
##done

## Integrated Bandpass Filter ---------------------------------------------------
##export QT_QPA_PLATFORM=offscreen

##SBATCH --ntasks=1
##SBATCH --cpus-per-task=12

## Grayscale
##for i in {0..4}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'conventional' --upsample_ratio 1 --maxiter 500 --binary 0 --eta 0.01 --brush_size 7
##done

##python /home/minseokhwan/Ensemble_Optimization/run_optimization_mode_converter.py --Nthreads 12 --n_seed 1 --load_data 0 --optimizer 'grayscale' --upsample_ratio 1 --maxiter 200 --brush_size 7

## Brush
##for i in {0..9}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'conventional' --upsample_ratio 1 --maxiter 500 --binary 1 --eta 0.001 --brush_size 7
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 32 --n_seed $i --load_data 1 --optimizer 'conventional' --upsample_ratio 1 --maxiter 500 --binary 1 --eta 0.001 --brush_size 7
##done

##python /home/minseokhwan/Ensemble_Optimization/run_optimization_mode_converter.py --Nthreads 12 --n_seed 9 --load_data 0 --optimizer 'brush' --upsample_ratio 1 --maxiter 250 --eta 0.001 --brush_size 7

## Ensemble
##for i in {0..9}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'ensemble' --upsample_ratio 1 --coeff_exp 20 --maxiter 500 --sigma_ensemble_max 0.01 --eta 0.0001 --brush_size 7
##done

##python /home/minseokhwan/Ensemble_Optimization/run_optimization_mode_converter.py --Nthreads 12 --n_seed 9 --load_data 0 --optimizer 'ensemble' --upsample_ratio 1 --coeff_exp 20 --maxiter 250 --sigma_ensemble_max 0.01 --eta 0.0001 --brush_size 7

##for i in 4 5 8 9;
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 48 --n_seed $i --load_data 0 --optimizer 'ensemble' --upsample_ratio 3 --coeff_exp 20 --maxiter 500 --sigma_ensemble_max 0.01 --eta 0.0001 --brush_size 7
##done

##python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 32 --n_seed 1 --load_data 0 --optimizer 'ensemble' --upsample_ratio 3 --coeff_exp 20 --maxiter 500 --sigma_ensemble_max 0.01 --eta 0.0001 --brush_size 7

## PSO
##for i in {3..9}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'PSO' --upsample_ratio 1 --maxiter 500 --brush_size 7
##done

##python /home/minseokhwan/Ensemble_Optimization/run_optimization_mode_converter.py --Nthreads 12 --n_seed 9 --load_data 0 --optimizer 'PSO' --upsample_ratio 1 --maxiter 250 --brush_size 7

## Mode Converter ---------------------------------------------------

##SBATCH --ntasks=48

## Grayscale
##for i in {0..4}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'conventional' --upsample_ratio 1 --maxiter 500 --binary 0 --eta 0.01 --brush_size 7
##done

##/opt/lumerical/v231/mpich2/nemesis/bin/mpiexec -n 32 python /home/minseokhwan/Ensemble_Optimization/run_optimization_mode_converter.py --Nthreads -1 --n_seed 1 --load_data 0 --optimizer 'conventional' --upsample_ratio 1 --maxiter 500 --binary 0 --eta 0.01 --brush_size 7

## Brush
##for i in {0..9}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'conventional' --upsample_ratio 1 --maxiter 500 --binary 1 --eta 0.001 --brush_size 7
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 32 --n_seed $i --load_data 1 --optimizer 'conventional' --upsample_ratio 1 --maxiter 500 --binary 1 --eta 0.001 --brush_size 7
##done

## Ensemble
##for i in {0..9}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'ensemble' --upsample_ratio 1 --coeff_exp 20 --maxiter 500 --sigma_ensemble_max 0.01 --eta 0.0001 --brush_size 7
##done

##/opt/lumerical/v231/mpich2/nemesis/bin/mpiexec -n 32 python /home/minseokhwan/Ensemble_Optimization/run_optimization_waveguide_U_bend.py --Nthreads -1 --n_seed 0 --load_data 0 --optimizer 'ensemble' --upsample_ratio 1 --coeff_exp 20 --maxiter 500 --sigma_ensemble_max 0.01 --eta 0.0001 --brush_size 7

##for i in 4 5 8 9;
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 48 --n_seed $i --load_data 0 --optimizer 'ensemble' --upsample_ratio 3 --coeff_exp 20 --maxiter 500 --sigma_ensemble_max 0.01 --eta 0.0001 --brush_size 7
##done

##python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 48 --n_seed 1 --load_data 0 --optimizer 'ensemble' --upsample_ratio 3 --coeff_exp 20 --maxiter 500 --sigma_ensemble_max 0.01 --eta 0.0001 --brush_size 7

## PSO
##for i in {3..9}
##do
##    python /home/minseokhwan/Ensemble_Optimization/run_optimization_integrated_bandpass_filter.py --Nthreads 32 --n_seed $i --load_data 0 --optimizer 'PSO' --upsample_ratio 1 --maxiter 500 --brush_size 7
##done