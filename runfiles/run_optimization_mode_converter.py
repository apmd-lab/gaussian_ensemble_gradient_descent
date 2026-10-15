import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
#sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')
#sys.path.append('/home/apmd/minseokhwan/gaussian_ensemble_gradient_descent')
#sys.path.append('/home/fs01/sm3266/gaussian_ensemble_gradient_descent')
sys.path.append('/u/smin2/gaussian_ensemble_gradient_descent')

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
parser.add_argument('--cuda_ind', type=int, default=0)
parser.add_argument('--precision', type=str, default='float32')
args = parser.parse_args()

cuda_ind = args.cuda_ind
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_ind)
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
#os.environ["OMP_NUM_THREADS"] = str(args.Nthreads)
#os.environ["OPENBLAS_NUM_THREADS"] = str(args.Nthreads)
#os.environ["MKL_NUM_THREADS"] = str(args.Nthreads)
#os.environ["NUMEXPR_NUM_THREADS"] = str(args.Nthreads)

import numpy as np
from gegd.optimizer import TF_BFGS, AF_STE, GEGD, PSO, GA, sep_CMA_ES, CMA_ES
import time

optimization_algorithm = args.optimizer
Nthreads = args.Nthreads
Nensemble = args.Nensemble
maxiter = args.maxiter if args.maxiter > 0 else None
n_seed = args.n_seed
load_data = args.load_data
cuda_ind = 0

# Geometry
Nx = args.Nx
Ny = args.Ny
symmetry = args.symmetry # Currently supported: (None), (D1,2,4)
periodic = 0
brush_size = args.brush_size # minimum feature size in pixels
upsample_ratio = args.upsample_ratio

# Define Cost Object
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import FDTD_functions.mode_converter_ceviche as objfun

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

cost_obj_high_fidelity = objfun.custom_objective(
    lam_tgt, # in um
    design_dim, # in um
    Nx*upsample_ratio,
    Ny*upsample_ratio,
    waveguide_width, # in um
    mat_padding,
    mat_waveguide,
)

cost_obj_low_fidelity = objfun.custom_objective(
    lam_tgt, # in um
    design_dim, # in um
    Nx*upsample_ratio,
    Ny*upsample_ratio,
    waveguide_width, # in um
    mat_padding,
    mat_waveguide,
)

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
low_fidelity_setting = 2 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 5 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
t_low_fidelity = 0.18 # low-fidelity simulation time in seconds
t_high_fidelity = 1.45 # high-fidelity simulation time in seconds
t_iteration = t_high_fidelity*Nensemble # target time per optimization iteration in seconds (actual time may be slightly longer due to the brush generator)
t_fwd_AD = 1.61

cost_obj_high_fidelity.set_accuracy(high_fidelity_setting)
cost_obj_low_fidelity.set_accuracy(low_fidelity_setting)

if optimization_algorithm == 'TF_BFGS':
    Ntrial = Nensemble #int(np.round(Nensemble/(t_fwd_AD/t_high_fidelity)))

    output_filename = 'polarization_beamsplitter_IPR' + str(int(1/IPR_exponent)) + '_Ntrial' + str(Ntrial) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
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
        cost_obj=cost_obj_high_fidelity,
        Nthreads=Nthreads,
    )
    
    T1 = time.time()
    optimizer.run(n_seed, output_filename, maxiter, beta_init=8.0, beta_ratio=2.0, n_beta=5, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'AF_STE':
    Ntrial = Nensemble #int(np.round(Nensemble/(t_fwd_AD/t_high_fidelity)))
    eta = args.eta

    output_filename = 'polarization_beamsplitter_IPR' + str(int(1/IPR_exponent)) + '_Ntrial' + str(Ntrial) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
        + '_eta' + str(eta) + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

    optimizer = AF_STE.optimizer(
        Nx=Nx,
        Ny=Ny,
        Ntrial=Ntrial,
        symmetry=symmetry,
        periodic=periodic,
        padding=padding,
        high_fidelity_setting=high_fidelity_setting,
        min_feature_size=min_feature_size,
        upsample_ratio=upsample_ratio,
        cost_obj=cost_obj_high_fidelity,
        Nthreads=Nthreads,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, maxiter, eta_ADAM=eta, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'GEGD':
    sigma_RBF = min_feature_size/2/np.sqrt(2)
    sigma_ensemble = args.sigma_ensemble # sampling standard deviation for the ensemble
    beta_proj = 8.0
    eta = args.eta
    coeff_exp = args.coeff_exp
    cost_threshold = 0.0

    output_filename = 'polarization_beamsplitter_IPR' + str(int(1/IPR_exponent)) + '_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
        + '_sig_ens' + str(sigma_ensemble) + '_eta' + str(eta) + '_mfs' + str(min_feature_size) + '_exp' + str(coeff_exp) + '_try' + str(n_seed+1)

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
        min_feature_size=min_feature_size,
        sigma_RBF=sigma_RBF,
        sigma_ensemble=sigma_ensemble,
        upsample_ratio=upsample_ratio,
        beta_proj=beta_proj,
        feasible_design_generation_method=feasible_design_generation_method,
        covariance_type='gaussian_constant', #'constant', 'gaussian_constant',
        coeff_exp=coeff_exp,
        cost_threshold=cost_threshold,
        cost_obj_high_fidelity=cost_obj_high_fidelity,
        cost_obj_low_fidelity=cost_obj_low_fidelity,
        use_ctrlVar=True,
        Nthreads=Nthreads,
        cuda_ind=cuda_ind,
        verbosity=1,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, eta_mu=eta, eta_sigma=eta, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'PSO':
    coeff_cognitive = 1.49
    coeff_social = 1.49
    coeff_inertia = 0.9

    output_filename = 'polarization_beamsplitter_IPR' + str(int(1/IPR_exponent)) + '_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
        + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

    optimizer = PSO.optimizer(
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
        cost_obj=cost_obj_high_fidelity,
        Nthreads=Nthreads,
        cuda_ind=cuda_ind,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, coeff_cognitive=coeff_cognitive, coeff_social=coeff_social, coeff_inertia=coeff_inertia, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'GA':
    output_filename = 'polarization_beamsplitter_IPR' + str(int(1/IPR_exponent)) + '_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
        + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

    optimizer = GA.optimizer(
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
        cost_obj=cost_obj_high_fidelity,
        Nthreads=Nthreads,
        cuda_ind=cuda_ind,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'sep_CMA_ES':
    output_filename = 'polarization_beamsplitter_IPR' + str(int(1/IPR_exponent)) + '_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
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
        cost_obj=cost_obj_high_fidelity,
        Nthreads=Nthreads,
        cuda_ind=cuda_ind,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)

elif optimization_algorithm == 'CMA_ES':
    output_filename = 'polarization_beamsplitter_IPR' + str(int(1/IPR_exponent)) + '_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
        + '_mfs' + str(min_feature_size) + '_try' + str(n_seed+1)

    optimizer = CMA_ES.optimizer(
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
        cost_obj=cost_obj_high_fidelity,
        Nthreads=Nthreads,
        cuda_ind=cuda_ind,
    )

    T1 = time.time()
    optimizer.run(n_seed, output_filename, load_data=load_data)
    T2 = time.time()
    print('\n### Total time: ' + str(T2 - T1), flush=True)