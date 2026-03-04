import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')

import numpy as np
import torch
from gegd.optimizer import TF_BFGS, AF_STE, GEGD, AF_PSO, AF_GA, sep_CMA_ES
from itertools import product
import time
import util.read_mat_data as rmd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Nthreads', type=int, default=1)
parser.add_argument('--n_seed', type=int, default=0)
parser.add_argument('--load_data', type=int, default=0)
parser.add_argument('--optimizer', type=str, default='GEGD')
parser.add_argument('--Nensemble', type=int, default=10)
parser.add_argument('--Nx', type=int, default=90)
parser.add_argument('--Ny', type=int, default=90)
parser.add_argument('--symmetry', type=int, default=0)
parser.add_argument('--upsample_ratio', type=int, default=1)
parser.add_argument('--coeff_exp', type=int, default=5)
parser.add_argument('--maxiter', type=int, default=100)
parser.add_argument('--sigma_ensemble', type=float, default=0.01)
parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--min_feature_size', type=int, default=7)
args = parser.parse_args()

optimization_algorithm = args.optimizer
Nthreads = args.Nthreads
Nensemble = args.Nensemble
maxiter = args.maxiter
n_seed = args.n_seed
load_data = args.load_data
cuda_ind = 0

# Geometry
Nx = args.Nx
Ny = args.Ny
symmetry = args.symmetry # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
min_feature_size = args.min_feature_size
upsample_ratio = args.upsample_ratio
feasible_design_generation_method = 'brush' # brush / two_phase_projection

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

cost_obj = objfun.custom_objective(cuda_ind=cuda_ind,
                                   symmetry=symmetry,
                                   periodic=periodic,
                                   Nx=Nx,
                                   Ny=Ny,
                                   Ndim=Ndim,
                                   min_feature_size=min_feature_size,
                                   feasible_design_generation_method=feasible_design_generation_method,
                                   n_seed=200,
                                   N_minima=1,
                                   scale=np.array([1.0, 0.0]*1),
                                   grayscale=False)

# Cost Function Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
low_fidelity_setting = 0.01 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 0.0 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)

t_low_fidelity = 1 # low-fidelity simulation time in seconds
t_high_fidelity = 10 # high-fidelity simulation time in seconds
t_iteration = t_high_fidelity*Nensemble #15.88 # target time per optimization iteration in seconds (actual time may be slightly longer due to the brush generator)
t_fwd_AD = 15

if optimization_algorithm == 'TF_BFGS':
    Ntrial = int(np.round(Nensemble/(t_fwd_AD/t_high_fidelity)))

    output_filename = 'test_Ntrial' + str(Ntrial) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

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
    sigma_RBF = min_feature_size/2/np.sqrt(2)
    sigma_ensemble = args.sigma_ensemble # sampling standard deviation for the ensemble
    beta_proj = 8.0
    eta = args.eta
    coeff_exp = args.coeff_exp
    cost_threshold = 0.0

    output_filename = 'test_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_sig_ens' + str(sigma_ensemble) \
        + '_eta' + str(eta) + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

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

    output_filename = 'test_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

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
    output_filename = 'test_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

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
    output_filename = 'test_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

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