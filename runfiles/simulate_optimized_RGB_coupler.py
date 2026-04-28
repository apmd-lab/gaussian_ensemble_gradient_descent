import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
#sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')
#sys.path.append('/home/apmd/minseokhwan/gaussian_ensemble_gradient_descent')
sys.path.append('/ocean/projects/cis260139p/smin2/gaussian_ensemble_gradient_descent/runfiles')

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import time

Nthreads = 8
cuda_ind = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_ind)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
jax.config.update("jax_enable_x64", True)

# Geometry
Nx = 60
Ny = 263
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
min_feature_size = 7 # minimum feature size in pixels
d_pixel = 0.01 # pixel side length (nm)
feasible_design_generation_method = 'brush' # brush / two_phase_projection
upsampling_ratio = 9

# Define Cost Object
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import RCWA_functions.RGB_coupler_FMMAX as objfun

IPR_exponent = 1/1
incident_pol = 'TM'

lam = np.array([0.675,0.540,0.450]) # um
theta_inc = np.array([0])*np.pi/180
phi_inc = np.array([0])*np.pi/180
in_plane_wavevector = np.array([0.0, 0.0])

diff_order = np.array([
    [0,4],
    [0,5],
    [0,6],
])
period = np.array([Nx * d_pixel, Ny * d_pixel])
thickness = 0.7

mat_pattern = np.array(['Air','TiO2_Sarkar']) # Low RI, High RI
mat_background = np.array(['Air','SiO2_bulk']) # background (incident side), background (exit side)

cost_obj = objfun.custom_objective(
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
    incident_pol=incident_pol,
)

cost_obj_upsampled = objfun.custom_objective(
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
    incident_pol=incident_pol,
)


# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
low_fidelity_setting = 24**2 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 50**2 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
cost_obj.set_accuracy(high_fidelity_setting)
cost_obj_upsampled.set_accuracy(high_fidelity_setting)

t1 = time.time()
print('### Running Simulations', flush=True)

print('\n\t*GEGD', end='', flush=True)
cost_all_GEGD = np.zeros(10)
for i in range(10):
    with np.load(directory + "/RCWA_functions/RGB_coupler/GEGD/RGB_coupler_IPR1_Nensemble20_Ndim60x263_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        cost_all_GEGD[i] = data['best_cost_hist'][-1]

idx_best = np.argmin(cost_all_GEGD)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD[idx_best], flush=True)

with np.load(directory + "/RCWA_functions/RGB_coupler/GEGD/RGB_coupler_IPR1_Nensemble20_Ndim60x263_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(idx_best + 1) + "_GEGD_results.npz") as data:
    x = data['best_x_final'].reshape(Nx, Ny)

T_tgt, E_in_plane, x_grid, y_grid, z_grid, transmitted_power, incident_power = cost_obj.get_diffraction_and_fields(x, 1)
x_up = np.where(gaussian_filter(zoom(x.astype(np.float64), upsampling_ratio, order=1, mode='wrap'), sigma=upsampling_ratio, mode='wrap') > 0.5, 1, 0)
T_tgt_up, E_in_plane_up, x_grid_up, y_grid_up, z_grid_up, transmitted_power_up, incident_power_up = cost_obj_upsampled.get_diffraction_and_fields(x_up, upsampling_ratio)

print('\n\t*BFGS', end='', flush=True)
cost_all_BFGS = np.zeros(180)
cost_all_BFGS_mfs = np.zeros(180)
for i in range(10):
    with np.load(directory + "/RCWA_functions/RGB_coupler/BFGS/RGB_coupler_IPR1_Ntrial18_Ndim60x263_D1_mfs7_try" + str(i + 1) + "_TF_results.npz") as data:
        cost_all_BFGS[18*i:18*(i+1)] = data['cost_fin'][0,:]
        cost_all_BFGS_mfs[18*i:18*(i+1)] = data['cost_fin'][1,:]

idx_best = np.argmin(cost_all_BFGS)
idx_best_mfs = np.argmin(cost_all_BFGS_mfs)
print(' --> Best Cost (idx=',idx_best+1,', idx_best_mfs=',idx_best_mfs,'): ',cost_all_BFGS[idx_best],' / ',cost_all_BFGS_mfs[idx_best_mfs],' (mfs not enforced / enforced)', flush=True)

print('\n\t*sep-CMA-ES', end='', flush=True)
cost_all_sep_CMA_ES = np.zeros(10)
for i in range(10):
    with np.load(directory + "/RCWA_functions/RGB_coupler/sep_CMA_ES/RGB_coupler_IPR1_Nensemble20_Ndim60x263_D1_mfs7_try" + str(i + 1) + "_sep_CMA_ES_results.npz") as data:
        cost_all_sep_CMA_ES[i] = data['best_cost_hist'][-1]

idx_best = np.argmin(cost_all_sep_CMA_ES)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_sep_CMA_ES[idx_best], flush=True)

print('\n\t*GA', end='', flush=True)
cost_all_GA = np.zeros(10)
for i in range(10):
    with np.load(directory + "/RCWA_functions/RGB_coupler/GA/RGB_coupler_IPR1_Nensemble20_Ndim60x263_D1_mfs7_try" + str(i + 1) + "_AF_GA_results.npz") as data:
        cost_all_GA[i] = data['best_cost_hist'][-1]

idx_best = np.argmin(cost_all_GA)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GA[idx_best], flush=True)

print('\n\t*AF-STE', end='', flush=True)
cost_all_AF_STE = np.zeros(180)
for i in range(10):
    with np.load(directory + "/RCWA_functions/RGB_coupler/AF_STE/RGB_coupler_IPR1_Ntrial18_Ndim60x263_D1_eta0.001_mfs7_try" + str(i + 1) + "_AF_STE_results.npz") as data:
        cost_all_AF_STE[18*i:18*(i+1)] = np.min(data['cost_hist'], axis=0)

idx_best = np.argmin(cost_all_AF_STE)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_AF_STE[idx_best], flush=True)

np.savez(directory + '/RCWA_functions/RGB_coupler/RGB_coupler_IPR1_Nensemble20_Ndim60x263_D1_mfs7_simulations',
    cost_all_GEGD=cost_all_GEGD,
    cost_all_BFGS=cost_all_BFGS,
    cost_all_BFGS_mfs=cost_all_BFGS_mfs,
    cost_all_sep_CMA_ES=cost_all_sep_CMA_ES,
    cost_all_GA=cost_all_GA,
    cost_all_AF_STE=cost_all_AF_STE,
    x=x,
    x_up=x_up,
    T_tgt=T_tgt,
    E_in_plane=E_in_plane,
    x_grid=x_grid,
    y_grid=y_grid,
    z_grid=z_grid,
    transmitted_power=transmitted_power,
    incident_power=incident_power,
    T_tgt_up=T_tgt_up,
    E_in_plane_up=E_in_plane_up,
    x_grid_up=x_grid_up,
    y_grid_up=y_grid_up,
    z_grid_up=z_grid_up,
    transmitted_power_up=transmitted_power_up,
    incident_power_up=incident_power_up,
)

t2 = time.time()
print('>>> Time taken: ' + str(np.round(t2 - t1, 2)) + ' s', flush=True)