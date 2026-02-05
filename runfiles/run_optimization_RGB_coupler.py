import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/apmd/minseokhwan/gaussian_ensemble_gradient_descent')

import numpy as np
import torch
from gegd.optimizer import TF_BFGS, GEGD, AF_PSO, AF_GA, sep_CMA_ES, AF_BL
from itertools import product
import time
import util.read_mat_data as rmd

Nthreads = 1
cuda_ind = 1

optimization_algorithm = 'GEGD' # TF_BFGS, AF_BL, GEGD, AF_GA, sep_CMA_ES, AF_PSO
n_seed = 0
Nensemble = 20
maxiter = 300
load_data = False

# Geometry
Nx = 37
Ny = 269
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
min_feature_size = 9 # minimum feature size in pixels
d_pixel = 0.008 # pixel side length (nm)
feasible_design_generation_method = 'two_phase_projection' # brush / two_phase_projection
upsample_ratio = 1

# Define Cost Object
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import RCWA_functions.RGB_coupler as objfun

IPR_exponent = 1/5

lam = np.array([0.675,0.540,0.450]) # um
theta_inc = np.array([0])*np.pi/180
phi_inc = np.array([0])*np.pi/180
angle_inc = np.array(list(product(theta_inc, phi_inc)))

theta_tgt = 45*np.pi/180
diff_order = np.array([
    [0,4],
    [0,5],
    [0,6],
])
period = np.array([Nx * d_pixel, Ny * d_pixel])
thickness = 0.7

mat_pattern = np.array(['Air','TiO2_Sarkar']) # Low RI, High RI
mat_background = np.array(['Air','SiO2_bulk']) # background (incident side), background (exit side)

cost_obj = objfun.custom_objective(mat_background,
                                   mat_pattern,
                                   Nthreads,
                                   diff_order,
                                   IPR_exponent=IPR_exponent,
                                   cuda_ind=cuda_ind)

cost_obj.set_geometry(Nx*upsample_ratio, Ny*upsample_ratio, period, thickness)
cost_obj.set_source(lam=lam, angle_inc=angle_inc)

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
low_fidelity_setting = [5,14] # [5,14], [5,15] # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = [8,20] # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
t_low_fidelity = 4.84 #4.84 5.75 # low-fidelity simulation time in seconds
t_high_fidelity = 34.8 # high-fidelity simulation time in seconds
t_iteration = t_high_fidelity*Nensemble #15.88 # target time per optimization iteration in seconds (actual time may be slightly longer due to the brush generator)
t_fwd_AD = 41.0

if optimization_algorithm == 'TF_BFGS':
    Ntrial = int(np.round(Nensemble/(t_fwd_AD/t_high_fidelity)))

    output_filename = 'RGB_coupler_IPR' + str(int(1/IPR_exponent)) + '_Ntrial' + str(Ntrial) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
        + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

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
    )
    
    T1 = time.time()
    optimizer.run(n_seed, output_filename, maxiter, beta_init=8.0, beta_ratio=2.0, n_beta=5, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'AF_BL':
    Ntrial = int(np.round(Nensemble/(t_fwd_AD/t_high_fidelity)))
    eta = 1e-2

    output_filename = 'RGB_coupler_IPR' + str(int(1/IPR_exponent)) + '_Ntrial' + str(Ntrial) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_eta' + str(eta) \
        + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

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
    sigma_RBF = min_feature_size/8
    sigma_ensemble = 1e-6 # sampling standard deviation for the ensemble
    beta_proj = 16.0
    eta = 1e-9
    coeff_exp = 20.0
    cost_threshold = 0.0

    output_filename = 'RGB_coupler_IPR' + str(int(1/IPR_exponent)) + '_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
        + '_sig_RBF' + str(sigma_RBF) + '_sig_ens' + str(sigma_ensemble) + '_beta_proj' + str(beta_proj) + '_eta' + str(eta) + '_mfs' + str(min_feature_size) \
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
        sigma_RBF=sigma_RBF,
        sigma_ensemble=sigma_ensemble,
        upsample_ratio=upsample_ratio,
        beta_proj=beta_proj,
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
    optimizer.run(n_seed, output_filename, eta_mu=eta, eta_sigma=eta, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'AF_PSO':
    coeff_cognitive = 1.49
    coeff_social = 1.49
    coeff_inertia = 0.9

    output_filename = 'RGB_coupler_IPR' + str(int(1/IPR_exponent)) + '_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
        + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

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
    output_filename = 'RGB_coupler_IPR' + str(int(1/IPR_exponent)) + '_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
        + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

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

elif optimization_algorithm == 'sep_CMA_ES':
    output_filename = 'RGB_coupler_IPR' + str(int(1/IPR_exponent)) + '_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
        + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

    optimizer = sep_CMA_ES.optimizer(
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