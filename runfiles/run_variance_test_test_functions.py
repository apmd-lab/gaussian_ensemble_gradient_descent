import os
directory = os.path.dirname(os.path.realpath(__file__))

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import optimizer.ensemble_optimization as ENSEMBLE
import optimizer.brush_generator as brush
from itertools import product
import time
import util.read_mat_data as rmd
import optimizer.density_transforms as dtf
import optimizer.symmetry_operations as symOp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Nthreads', type=int, default=1)
parser.add_argument('--upsample_ratio', type=int, default=1)
parser.add_argument('--brush_size', type=int, default=7)
args = parser.parse_args()

Nthreads = args.Nthreads

# Geometry
Nx = 35
Ny = 70
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
brush_size = args.brush_size # minimum feature size in pixels
maxiter = 500 # total number of iterations
low_fidelity_setting = 0.001 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 0.0 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
upsample_ratio = args.upsample_ratio

sigma_ensemble_max = 0.01 # maximum sampling standard deviation for the ensemble
covariance_type = 'gaussian_constant' # structure of the covariance of the multivariate normal sampling distribution (constant, diagonal, gaussian_constant, gaussian_diagonal)
t_low_fidelity = 1 # low-fidelity simulation time in seconds
t_high_fidelity = 10 # high-fidelity simulation time in seconds
t_iteration = t_high_fidelity*10 #15.88 # target time per optimization iteration in seconds (actual time may be slightly longer due to the brush generator)
Nensemble = 1000 # only for ensemble
r_CV = 1
eta_mu = 0.0001 # NES: 0.01, ADAM: 0.01
eta_sigma = 0.0001 # NES: 1.0, ADAM: 0.1
coeff_exp = 20
cost_threshold = 0.0

# Define Cost Object
#----------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, save_t_array=False) --> return cost
# (2) set_accuracy(n_harmonic)
#----------------------------------------------------------
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

#x0 = np.zeros(Ndim)
#sigma_ensemble = sigma_ensemble_max/2*np.ones(1)
#
#with np.load(directory + '/results/ensemble_ADAM_results_x_hist_fin_test_Ndim35x70_D1_sigma0.01_coeffExp20_brush7_try2.npz') as data:
#    dx1 = data['x_latent_hist'][-1,:]
#    dx2 = data['x_latent_hist'][284,:]
#
## Compute Orthonormal Basis
#dx1 = dx1/np.linalg.norm(dx1)
#dx2_proj = dx2 - np.dot(dx1, dx2)*dx1
#dx2 = dx2_proj/np.linalg.norm(dx2_proj)
#
#zoom = 2
#
#suffix = 'test_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_Nensemble' + str(Nensemble) + '_rCV' + str(r_CV) + '_sigma' + str(sigma_ensemble_max) + '_exp' + str(coeff_exp)\
#         + '_brush' + str(brush_size) + '_zoom' + str(zoom) # + '_iter' + str(n_iter)
#
#T1 = time.time()
#optimizer.sample_cost_fct_landscape(suffix, Nensemble, r_CV, sigma_ensemble, dx1, dx2, 101, 1801, 21, 361, read_save_data=False, zoom=zoom)
#T2 = time.time()
#print('\n### Total time: ' + str(T2 - T1), flush=True)

n_iter = 100 #100-130

with np.load(directory + '/results/diffraction_grating/test_rbf/ensemble_ADAM_results_x_hist_fin_test_Ndim35x70_D1_sigma0.01_coeffExp20_brush7_try2.npz') as data:
    x0 = data['x_latent_hist'][n_iter,:]

suffix = 'test_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_Nensemble' + str(Nensemble) + '_sigma' + str(sigma_ensemble_max) + '_exp' + str(coeff_exp)\
         + '_brush' + str(brush_size) + '_iter' + str(n_iter)

Nensemble = 100
r_CV_list = np.array([1,5,10,20])
Nsample = 1000

T1 = time.time()
optimizer.test_CV(suffix, Nensemble, r_CV_list, Nsample, x0, sigma_ensemble_max, read_save_data=False)
T2 = time.time()
print('\n### Total time: ' + str(T2 - T1), flush=True)