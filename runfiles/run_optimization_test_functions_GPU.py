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

# Geometry
Nx = 90
Ny = 90
symmetry = 0 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
upsample_ratio = 1

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
feasible_design_generation_method = 'two_phase_projection' # brush / two_phase_projection
min_feature_size = 7 # minimum feature size in pixels
maxiter = 1000 # total number of iterations
low_fidelity_setting = 0.001 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 0.0 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)

Nensemble = 10
sigma_ensemble_max = 1e-2 # maximum sampling standard deviation for the ensemble
t_low_fidelity = 1 # low-fidelity simulation time in seconds
t_high_fidelity = 10 # high-fidelity simulation time in seconds
t_iteration = t_high_fidelity*Nensemble #15.88 # target time per optimization iteration in seconds (actual time may be slightly longer due to the brush generator)

eta = 1e-1
var_ensemble = 0.001
asymmetry_factor = 0.01
coeff_exp = 20.0
cost_threshold = 0.0

t_fwd_AD = 15
Ntrial = int(np.round(10/(t_fwd_AD/t_high_fidelity))) # only for conventional

Nswarm = 40 # only for PSO, NES, GA
coeff_cognitive = 1.49
coeff_social = 1.49
coeff_inertia = 0.9

# Define Cost Object
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import test_functions.test_objective_function_rbf as objfun

if symmetry == 0:
    Ndim = Nx*Ny
   
elif symmetry == 1:
    Ndim = int(np.floor(Nx*upsample_ratio/2 + 0.5)*Ny*upsample_ratio)

elif symmetry == 2:
    Ndim = int(np.floor(Nx*upsample_ratio/2 + 0.5)*np.floor(Ny*upsample_ratio/2 + 0.5))

elif symmetry == 4:
    Ndim = int(np.floor(Nx*upsample_ratio/2 + 0.5)*(np.floor(Nx*upsample_ratio/2 + 0.5) + 1)/2)

zoom_factor = 1.0
cost_obj = objfun.custom_objective(cuda_ind=0,
                                   symmetry=0,
                                   periodic=periodic,
                                   Nx=90,
                                   Ny=90,
                                   Ndim=90**2,
                                   min_feature_size=7,
                                   feasible_design_generation_method=feasible_design_generation_method,
                                   brush_shape='circle',
                                   n_seed=100,
                                   N_minima=10,
                                   scale=2.0,
                                   zoom_factor=zoom_factor)

n_seed = 0
optimization_algorithm = 'AF_GA'
load_data = False

output_filename = 'test_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_zoom' + str(zoom_factor) + '_eta' + str(eta) + '_mfs' + str(min_feature_size) \
    + '_var' + str(var_ensemble) + '_asym' + str(asymmetry_factor) + '_try' + str(n_seed+1)

if optimization_algorithm == 'TF_BFGS':
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

elif optimization_algorithm == 'GEGD':
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
        sigma_ensemble_max=sigma_ensemble_max,
        upsample_ratio=upsample_ratio,
        feasible_design_generation_method=feasible_design_generation_method,
        covariance_type='gaussian_constant',
        coeff_exp=coeff_exp,
        cost_threshold=cost_threshold,
        cost_obj=cost_obj,
        Nthreads=Nthreads,
        verbosity=1,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, eta_mu=eta, eta_sigma=eta, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'ACMA_ES':
    optimizer = ACMA_ES.optimizer(
        Nx=Nx,
        Ny=Ny,
        Nsample=Nswarm,
        symmetry=symmetry,
        periodic=periodic,
        padding=padding,
        maxiter=maxiter,
        high_fidelity_setting=high_fidelity_setting,
        min_feature_size=min_feature_size,
        sigma_RBF=min_feature_size/2,
        upsample_ratio=upsample_ratio,
        feasible_design_generation_method=feasible_design_generation_method,
        cost_obj=cost_obj,
        Nthreads=Nthreads,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, eta=eta, cuda_ind=0, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'D_GEGD':
    output_filename = 'test_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_zoom' + str(zoom_factor) + '_eta' + str(eta) + '_mfs' + str(min_feature_size) \
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
        cuda_ind=0,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, s_mag0=1e-2, eta=eta, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'AF_PSO':
    optimizer = AF_PSO.optimizer(
        Nx=Nx,
        Ny=Ny,
        Nswarm=Nswarm,
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
        cuda_ind=0,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, coeff_cognitive=coeff_cognitive, coeff_social=coeff_social, coeff_inertia=coeff_inertia, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'AF_GA':
    output_filename = 'test_Nswarm' + str(Nswarm) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_zoom' + str(zoom_factor) + '_mfs' + str(min_feature_size) \
        + '_try' + str(n_seed+1)

    optimizer = AF_GA.optimizer(
        Nx=Nx,
        Ny=Ny,
        Nbatch=Nswarm,
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
        cuda_ind=0,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'AF_CMA_ES':
    output_filename = 'test_Nswarm' + str(Nswarm) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_zoom' + str(zoom_factor) + '_mfs' + str(min_feature_size) \
        + '_try' + str(n_seed+1)

    optimizer = AF_CMA_ES.optimizer(
        Nx=Nx,
        Ny=Ny,
        Nsample=Nswarm,
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
        cuda_ind=0,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)