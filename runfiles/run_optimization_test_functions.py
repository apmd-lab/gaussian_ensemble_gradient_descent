import os
directory = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import torch
import optimizer.ensemble_optimization as ENSEMBLE
import optimizer.grayscale_optimization as GRAY
import optimizer.brush_optimization as BRUSH
import optimizer.PSO as PSO
from itertools import product
import time
import util.read_mat_data as rmd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Nthreads', type=int, default=1)
parser.add_argument('--n_seed', type=int, default=0)
parser.add_argument('--load_data', type=int, default=0)
parser.add_argument('--optimizer', type=str, default='ensemble')
parser.add_argument('--upsample_ratio', type=int, default=1)
parser.add_argument('--coeff_exp', type=int, default=5)
parser.add_argument('--maxiter', type=int, default=100)
parser.add_argument('--sigma_ensemble_max', type=float, default=0.01)
parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--brush_size', type=int, default=7)
args = parser.parse_args()

Nthreads = args.Nthreads

# Geometry
Nx = 35
Ny = 70
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
upsample_ratio = args.upsample_ratio

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
brush_size = 7 # minimum feature size in pixels
maxiter = args.maxiter # total number of iterations
low_fidelity_setting = 0.001 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 0.0 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
upsample_ratio = args.upsample_ratio

sigma_ensemble_max = args.sigma_ensemble_max # maximum sampling standard deviation for the ensemble
covariance_type = 'constant' # structure of the covariance of the multivariate normal sampling distribution (constant, diagonal, gaussian_constant, gaussian_diagonal)
t_low_fidelity = 1 # low-fidelity simulation time in seconds
t_high_fidelity = 10 # high-fidelity simulation time in seconds
t_iteration = t_high_fidelity*10 #15.88 # target time per optimization iteration in seconds (actual time may be slightly longer due to the brush generator)
#Nensemble = 5 # only for ensemble
#r_CV = int(np.round((t_iteration - Nensemble*t_high_fidelity)/(Nensemble*t_low_fidelity)))
eta_mu = args.eta # NES: 0.01, ADAM: 0.01
eta_sigma = 0.0001 # NES: 1.0, ADAM: 0.1
coeff_exp = args.coeff_exp
cost_threshold = 0.0

t_fwd_AD = 15
Ntrial = int(np.round(10/(t_fwd_AD/t_high_fidelity))) # only for conventional
eta_ADAM = args.eta # 0.01 (grayscale), 0.001 (brush)

Nswarm = 10 # only for PSO
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

cost_obj = objfun.custom_objective(cuda_ind=0,
                                   symmetry=symmetry,
                                   periodic=periodic,
                                   Nx=Nx*upsample_ratio,
                                   Ny=Ny*upsample_ratio,
                                   Ndim=Ndim,
                                   brush_size=brush_size*upsample_ratio,
                                   brush_shape='circle',
                                   n_seed=100,
                                   N_minima=10,
                                   scale=2.0)

n_seed = args.n_seed
optimization_algorithm = args.optimizer # conventional / ensemble / PSO

suffix = 'test_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_sigma' + str(sigma_ensemble_max) + '_coeffExp' + str(coeff_exp) + '_brush' + str(brush_size) + '_try' + str(n_seed+1)

if optimization_algorithm == 'grayscale':
    optimizer = GRAY.optimizer(Nx=Nx,
                               Ny=Ny,
                               Ntrial=Ntrial,
                               symmetry=symmetry,
                               periodic=periodic,
                               padding=padding,
                               high_fidelity_setting=high_fidelity_setting,
                               brush_size=brush_size,
                               upsample_ratio=upsample_ratio,
                               cost_obj=cost_obj,
                               Nthreads=Nthreads)
    
    T1 = time.time()
    optimizer.run(n_seed, suffix, maxiter, beta_init=8.0, beta_ratio=2.0, n_beta=5, load_data=args.load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'brush':
    optimizer = BRUSH.optimizer(Nx=Nx,
                                Ny=Ny,
                                Ntrial=Ntrial,
                                symmetry=symmetry,
                                periodic=periodic,
                                padding=padding,
                                high_fidelity_setting=high_fidelity_setting,
                                brush_size=brush_size,
                                upsample_ratio=upsample_ratio,
                                cost_obj=cost_obj,
                                Nthreads=Nthreads)
    
    T1 = time.time()
    optimizer.run(n_seed, suffix, maxiter, eta_ADAM=eta_ADAM, load_data=args.load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'ensemble':
    optimizer = ENSEMBLE.optimizer(Nx=Nx,
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
                                   brush_size=brush_size,
                                   sigma_ensemble_max=sigma_ensemble_max,
                                   upsample_ratio=upsample_ratio,
                                   covariance_type=covariance_type,
                                   coeff_exp=coeff_exp,
                                   cost_threshold=cost_threshold,
                                   cost_obj=cost_obj,
                                   Nthreads=Nthreads)

    T1 = time.time()
    optimizer.run(n_seed, suffix, optimizer='ADAM', eta_mu=eta_mu, eta_sigma=eta_sigma, load_data=args.load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'PSO':
    optimizer = PSO.optimizer(Nx=Nx,
                              Ny=Ny,
                              Nswarm=Nswarm,
                              symmetry=symmetry,
                              periodic=periodic,
                              padding=padding,
                              maxiter=maxiter,
                              high_fidelity_setting=high_fidelity_setting,
                              brush_size=brush_size,
                              upsample_ratio=upsample_ratio,
                              cost_obj=cost_obj,
                              Nthreads=Nthreads)

    T1 = time.time()
    optimizer.run(n_seed, suffix, coeff_cognitive=coeff_cognitive, coeff_social=coeff_social, coeff_inertia=coeff_inertia, load_data=args.load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)