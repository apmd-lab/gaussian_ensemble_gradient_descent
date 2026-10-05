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
from gegd.optimizer import GEGD
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
periodic = 1
padding = None
min_feature_size = args.min_feature_size
d_pixel = 0.01 # pixel side length (nm)
feasible_design_generation_method = 'brush' # brush / two_phase_projection
upsample_ratio = args.upsample_ratio

# Define Cost Object
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import RCWA_functions.polarization_beamsplitter_FMMAX as objfun

IPR_exponent = 1/1

lam = np.array([0.633]) # um
theta_inc = np.array([0])*np.pi/180
phi_inc = np.array([0])*np.pi/180
in_plane_wavevector = np.array([0.0, 0.0])

diff_order = np.array([
    [0,-1],
    [0,1],
])
period = np.array([Nx * d_pixel, Ny * d_pixel])
thickness = 0.3

mat_pattern = np.array(['Air','Si_Schinke_Shkondin']) # Low RI, High RI
mat_background = np.array(['SiO2_bulk','Air']) # background (incident side), background (exit side)

cost_obj_high_fidelity = objfun.custom_objective(
    Nx,
    Ny,
    period,
    thickness,
    lam,
    in_plane_wavevector,
    mat_background,
    mat_pattern,
    diff_order,
    IPR_exponent=IPR_exponent,
    precision=args.precision,
)

cost_obj_low_fidelity = objfun.custom_objective(
    Nx,
    Ny,
    period,
    thickness,
    lam,
    in_plane_wavevector,
    mat_background,
    mat_pattern,
    diff_order,
    IPR_exponent=IPR_exponent,
    precision=args.precision,
)

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
low_fidelity_setting = 18**2 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 36**2 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
t_low_fidelity = 0.18 # low-fidelity simulation time in seconds
t_high_fidelity = 1.45 # high-fidelity simulation time in seconds
t_iteration = t_high_fidelity*Nensemble # target time per optimization iteration in seconds (actual time may be slightly longer due to the brush generator)
t_fwd_AD = 1.61

cost_obj_high_fidelity.set_accuracy(high_fidelity_setting)
cost_obj_low_fidelity.set_accuracy(low_fidelity_setting)

sigma_RBF = min_feature_size/2/np.sqrt(2)
sigma_ensemble = args.sigma_ensemble # sampling standard deviation for the ensemble
beta_proj = 8.0
eta = args.eta
coeff_exp = args.coeff_exp
cost_threshold = 0.0

output_filename = 'polarization_beamsplitter_ensemble_gradient_IPR' + str(int(1/IPR_exponent)) + '_Nensemble' + str(Nensemble) + '_Ndim' + str(Nx) + 'x' + str(Ny) + '_D' + str(symmetry) \
    + '_sig_ens' + str(sigma_ensemble) + '_eta' + str(eta) + '_mfs' + str(min_feature_size) + '_exp' + str(coeff_exp) + '_try' + str(n_seed+1)

T1 = time.time()
np.random.seed(n_seed)

# Without Control Variates
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
    use_ctrlVar=False,
    Nthreads=Nthreads,
    cuda_ind=cuda_ind,
    verbosity=1,
)

jac_noCV = np.zeros((optimizer.Ndim, 10))
for i in range(10):
    _, _, _, _, _, _, jac_noCV[:,i], _, _, _, _, _, _, _ = optimizer.ensemble_jacobian(np.zeros(optimizer.Ndim), False)

# With Control Variates
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

jac_CV = np.zeros((optimizer.Ndim, 10))
for i in range(10):
    _, _, _, _, _, _, jac_CV[:,i], _, _, _, _, _, _, _ = optimizer.ensemble_jacobian(np.zeros(optimizer.Ndim), False)

cos_sim = jac_noCV.T @ jac_CV / (np.linalg.norm(jac_noCV, axis=0)[:,None] * np.linalg.norm(jac_CV, axis=0)[None,:])

np.savez(
    output_filename + '.npz',
    jac_noCV=jac_noCV,
    jac_CV=jac_CV,
    cos_sim=cos_sim,
)

T2 = time.time()
print('\n### Total time: ' + str(T2 - T1), flush=True)