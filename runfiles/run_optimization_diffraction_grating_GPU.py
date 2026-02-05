import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/apmd/minseokhwan/gaussian_ensemble_gradient_descent')

import numpy as np
import torch
from gegd.optimizer import TF_BFGS, AF_STE, GEGD, AF_PSO, AF_GA, AF_CMA_ES, AF_BL, ACMA_ES, D_GEGD
from itertools import product
import time
import util.read_mat_data as rmd

Nthreads = 1
cuda_ind = 0
optimization_algorithm = 'GEGD' # TF_BFGS, AF_GA, AF_CMA_ES, AF_PSO, D_GEGD
load_data = False

# Geometry
Nx = 80
Ny = 160
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
upsample_ratio = 1

# Define Cost Object
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import RCWA_functions.diffraction_grating as objfun

lam = np.array([0.633]) # um
theta_inc = np.array([0])*np.pi/180
phi_inc = np.array([0])*np.pi/180
angle_inc = np.array(list(product(theta_inc, phi_inc)))

theta_tgt = 45*np.pi/180
max_diff_order = np.array([0,1])
Ly = max_diff_order[1]*np.max(lam)/np.sin(theta_tgt)
period = np.array([Ly*Nx/Ny,Ly])
thickness = 0.3

mat_multilayer = np.array(['Si_Schinke_Shkondin']) # Top to bottom / Si3N4_Luke / Si_Schinke_Shkondin
mat_background = np.array(['SiO2_bulk','Air']) # background (incident side), background (exit side)

cost_obj = objfun.custom_objective(mat_background,
                                   mat_multilayer,
                                   Nthreads,
                                   minimax=False,
                                   cuda_ind=cuda_ind)#(args.optimizer=='ensemble' and upsample_ratio > 1))

cost_obj.set_geometry(Nx*upsample_ratio, Ny*upsample_ratio, period, thickness)
cost_obj.set_source(lam=lam, angle_inc=angle_inc)

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
feasible_design_generation_method = 'two_phase_projection' # brush / two_phase_projection
min_feature_size = 9 # minimum feature size in pixels
maxiter = 200 # total number of iterations
low_fidelity_setting = [4,8] # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = [9,18] # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)

Nensemble = 10
t_low_fidelity = 0.317 #0.17 0.31 # low-fidelity simulation time in seconds
t_high_fidelity = 8.99 # high-fidelity simulation time in seconds
t_iteration = t_high_fidelity*Nensemble #15.88 # target time per optimization iteration in seconds (actual time may be slightly longer due to the brush generator)

eta = 1e-1
var_ensemble = 0.001
asymmetry_factor = 0.0
coeff_exp = 20.0
cost_threshold = 0.0

t_fwd_AD = 11.0
Ntrial = int(np.round(Nensemble/(t_fwd_AD/t_high_fidelity))) # only for conventional

coeff_cognitive = 1.49
coeff_social = 1.49
coeff_inertia = 0.9

for n_seed in range(0, 10):
    output_filename = 'test_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_eta' + str(eta) + '_mfs' + str(min_feature_size) \
        + '_var' + str(var_ensemble) + '_asym' + str(asymmetry_factor) + '_try' + str(n_seed+1)

    if optimization_algorithm == 'TF_BFGS':
        output_filename = 'diffraction_grating_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_mfs' + str(min_feature_size) \
            + '_try' + str(n_seed+1)

        optimizer = TF_BFGS.optimizer(
            Nx=Nx,
            Ny=Ny,
            Ntrial=Ntrial,
            symmetry=symmetry,
            periodic=periodic,
            padding=padding,
            high_fidelity_setting=high_fidelity_setting,
            min_feature_size=min_feature_size,
            upsample_ratio=upsample_ratio,
            cost_obj=cost_obj,
            Nthreads=Nthreads,
            cuda_ind=cuda_ind,
        )
        
        T1 = time.time()
        optimizer.run(n_seed, output_filename, maxiter, beta_init=8.0, beta_ratio=2.0, n_beta=5, load_data=load_data)
        T2 = time.time()
        print('\n### Total time: ' + str(T2 - T1), flush=True)

    elif optimization_algorithm == 'AF_BL':
        output_filename = 'test_Ntrial' + str(Ntrial) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_eta' + str(eta) + '_mfs' + str(min_feature_size) \
            + '_try' + str(n_seed+1)

        optimizer = AF_BL.optimizer(
            Nx=Nx,
            Ny=Ny,
            Ntrial=Ntrial,
            symmetry=symmetry,
            periodic=periodic,
            padding=padding,
            maxiter=maxiter,
            high_fidelity_setting=high_fidelity_setting,
            min_feature_size=min_feature_size,
            upsample_ratio=upsample_ratio,
            cost_obj=cost_obj,
            Nthreads=Nthreads,
            cuda_ind=cuda_ind,
        )
        
        T1 = time.time()
        optimizer.run(n_seed, output_filename, eta=eta, load_data=load_data)
        T2 = time.time()
        print('\n### Total time: ' + str(T2 - T1), flush=True)

    elif optimization_algorithm == 'GEGD':
        output_filename = 'diffraction_grating_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_sigma' + str(1e-6) + '_eta' + str(1e-8) + '_mfs' + str(min_feature_size) \
            + '_try' + str(n_seed+1)

        optimizer = GEGD.optimizer(
            Nx=Nx,
            Ny=Ny,
            symmetry=symmetry,
            periodic=periodic,
            padding=padding,
            maxiter=maxiter,
            t_low_fidelity=t_low_fidelity,
            t_high_fidelity=t_high_fidelity,
            t_iteration=t_iteration,
            high_fidelity_setting=high_fidelity_setting,
            low_fidelity_setting=low_fidelity_setting,
            min_feature_size=min_feature_size,
            sigma_RBF=min_feature_size/2/np.sqrt(2),
            sigma_ensemble=1e-6,
            upsample_ratio=upsample_ratio,
            feasible_design_generation_method=feasible_design_generation_method,
            covariance_type='gaussian_constant',
            coeff_exp=coeff_exp,
            cost_threshold=cost_threshold,
            cost_obj=cost_obj,
            Nthreads=Nthreads,
            cuda_ind=cuda_ind,
            verbosity=1,
        )

        T1 = time.time()
        optimizer.run(n_seed, output_filename, eta_mu=1e-8, eta_sigma=1e-8, load_data=load_data)
        T2 = time.time()
        print('\n### Total time: ' + str(T2 - T1), flush=True)

    elif optimization_algorithm == 'D_GEGD':
        output_filename = 'diffraction_grating_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_eta' + str(eta) + '_mfs' + str(min_feature_size) \
            + '_var' + str(var_ensemble) + '_asym' + str(asymmetry_factor) + '_try' + str(n_seed+1)

        optimizer = D_GEGD.optimizer(
            Nx=Nx,
            Ny=Ny,
            symmetry=symmetry,
            periodic=periodic,
            padding=padding,
            maxiter=maxiter,
            t_low_fidelity=t_low_fidelity,
            t_high_fidelity=t_high_fidelity,
            t_iteration=t_iteration,
            high_fidelity_setting=high_fidelity_setting,
            low_fidelity_setting=low_fidelity_setting,
            min_feature_size=min_feature_size,
            sigma_RBF=min_feature_size/2/np.sqrt(2),
            var_ensemble=var_ensemble,
            asymmetry_factor=asymmetry_factor,
            upsample_ratio=upsample_ratio,
            feasible_design_generation_method=feasible_design_generation_method,
            coeff_exp=coeff_exp,
            cost_threshold=cost_threshold,
            cost_obj=cost_obj,
            Nthreads=Nthreads,
            verbosity=1,
            cuda_ind=cuda_ind,
        )

        T1 = time.time()
        optimizer.run(n_seed, output_filename, s_mag0=1e-2, eta=eta, load_data=load_data)
        T2 = time.time()
        print('\n### Total time: ' + str(T2 - T1), flush=True)

    elif optimization_algorithm == 'AF_PSO':
        output_filename = 'diffraction_grating_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_mfs' + str(min_feature_size) \
            + '_try' + str(n_seed+1)

        optimizer = AF_PSO.optimizer(
            Nx=Nx,
            Ny=Ny,
            Nswarm=Nensemble,
            symmetry=symmetry,
            periodic=periodic,
            padding=padding,
            maxiter=maxiter,
            high_fidelity_setting=high_fidelity_setting,
            min_feature_size=min_feature_size,
            upsample_ratio=upsample_ratio,
            feasible_design_generation_method=feasible_design_generation_method,
            cost_obj=cost_obj,
            Nthreads=Nthreads,
            cuda_ind=cuda_ind,
        )

        T1 = time.time()
        optimizer.run(n_seed, output_filename, coeff_cognitive=coeff_cognitive, coeff_social=coeff_social, coeff_inertia=coeff_inertia, load_data=load_data)
        T2 = time.time()
        print('\n### Total time: ' + str(T2 - T1), flush=True)

    elif optimization_algorithm == 'AF_GA':
        output_filename = 'diffraction_grating_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_mfs' + str(min_feature_size) \
            + '_try' + str(n_seed+1)

        optimizer = AF_GA.optimizer(
            Nx=Nx,
            Ny=Ny,
            Nbatch=Nensemble,
            symmetry=symmetry,
            periodic=periodic,
            padding=padding,
            maxiter=maxiter,
            high_fidelity_setting=high_fidelity_setting,
            min_feature_size=min_feature_size,
            upsample_ratio=upsample_ratio,
            feasible_design_generation_method=feasible_design_generation_method,
            cost_obj=cost_obj,
            Nthreads=Nthreads,
            cuda_ind=cuda_ind,
        )

        T1 = time.time()
        optimizer.run(n_seed, output_filename, load_data=load_data)
        T2 = time.time()
        print('\n### Total time: ' + str(T2 - T1), flush=True)

    elif optimization_algorithm == 'AF_CMA_ES':
        output_filename = 'diffraction_grating_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_mfs' + str(min_feature_size) \
            + '_try' + str(n_seed+1)

        optimizer = AF_CMA_ES.optimizer(
            Nx=Nx,
            Ny=Ny,
            Nsample=Nensemble,
            symmetry=symmetry,
            periodic=periodic,
            padding=padding,
            maxiter=maxiter,
            high_fidelity_setting=high_fidelity_setting,
            min_feature_size=min_feature_size,
            upsample_ratio=upsample_ratio,
            feasible_design_generation_method=feasible_design_generation_method,
            cost_obj=cost_obj,
            Nthreads=Nthreads,
            cuda_ind=cuda_ind,
        )

        T1 = time.time()
        optimizer.run(n_seed, output_filename, load_data=load_data)
        T2 = time.time()
        print('\n### Total time: ' + str(T2 - T1), flush=True)