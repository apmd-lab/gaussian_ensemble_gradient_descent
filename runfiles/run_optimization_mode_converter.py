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

from mpi4py import MPI
comm = MPI.COMM_WORLD

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
parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--brush_size', type=int, default=7)
args = parser.parse_args()

Nthreads = args.Nthreads

# Geometry
Nx = 70
Ny = 70
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 0
brush_size = args.brush_size # minimum feature size in pixels
upsample_ratio = args.upsample_ratio

# Define Cost Object
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import FDTD_functions.mode_converter_lumFDTD as objfun

c = 299792458
um = 1e-6
nm = 1e-9

lam_tgt = 1.55*um

design_dim = np.array([1*(Nx/Ny)*lam_tgt,1*lam_tgt,400*nm])
waveguide_width = 0.5*um

dpix = design_dim[0]/Nx
waveguide_width_npix = np.round(waveguide_width/dpix)
if waveguide_width_npix < brush_size:
    print('\t!!! Waveguide width smaller than the minimum feature size !!!', flush=True)
    assert False
if Nx % 2 == 0:
    waveguide_width_npix += (waveguide_width_npix % 2)*np.sign(waveguide_width/dpix - waveguide_width_npix)
else:
    waveguide_width_npix += (waveguide_width_npix % 2 == 0)*np.sign(waveguide_width/dpix - waveguide_width_npix)
waveguide_halfwidth_npix = int(np.floor(waveguide_width_npix/2 + 0.5))
center_x = int(np.floor(Nx/2 + 0.5))
waveguide_bottom = int(center_x - waveguide_halfwidth_npix)
waveguide_top = int(waveguide_bottom + waveguide_width_npix)

padding = -np.ones((Nx + 2*brush_size, Ny + 2*brush_size))
padding[brush_size:-brush_size,brush_size:-brush_size] = 0
padding[waveguide_bottom + brush_size:waveguide_top + brush_size,:brush_size] = 1
padding[waveguide_bottom + brush_size:waveguide_top + brush_size,-brush_size:] = 1

mat_padding = 'SiO2_bulk'
mat_waveguide = 'Si_Schinke_Shkondin'

fsp_suffix = args.optimizer + str(args.n_seed)

cost_obj = objfun.custom_objective(2,
                                   lam_tgt, # in m
                                   design_dim, # in m
                                   waveguide_width, # in m
                                   Nx,
                                   Ny,
                                   mat_padding,
                                   mat_waveguide,
                                   Nthreads,
                                   fsp_suffix)

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
maxiter = args.maxiter # total number of iterations
low_fidelity_setting = 2 #1 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 5 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)

sigma_ensemble_max = args.sigma_ensemble_max # sampling width for the ensemble (start with large value and decrease if sigma_ensemble increases during optimization)
covariance_type = 'gaussian_constant' # structure of the covariance of the multivariate normal sampling distribution (constant, diagonal, gaussian_constant, gaussian_diagonal)
t_low_fidelity = 1.0 # low-fidelity simulation time in seconds
t_high_fidelity = 3.6 #5.1 # high-fidelity simulation time in seconds
t_iteration = t_high_fidelity*20 # target time per optimization iteration in seconds (actual time may be slightly longer due to the brush generator)
eta_mu = args.eta
eta_sigma = 0.001
coeff_exp = args.coeff_exp
cost_threshold = 0

t_fwd_bwd = 7.1
Ntrial = int(np.round(20/(t_fwd_bwd/t_high_fidelity))) # only for grayscale / brush
eta_ADAM = args.eta # 0.01 (grayscale), 0.001 (brush)

Nswarm = 20 # only for PSO
coeff_cognitive = 1.49
coeff_social = 1.49
coeff_inertia = 0.9

n_seed = args.n_seed
optimization_algorithm = args.optimizer # grayscale / brush / ensemble / PSO

suffix = 'mode_converter_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) + '_upsample' + str(upsample_ratio) + '_sigma' + str(sigma_ensemble_max) + '_coeffExp' + str(coeff_exp) + '_brush' + str(brush_size) + '_try' + str(n_seed+1)

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
    if comm.rank == 0:
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
    if comm.rank == 0:
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
    if comm.rank == 0:
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
    if comm.rank == 0:
        print('\n### Total time: ' + str(T2 - T1), flush=True)