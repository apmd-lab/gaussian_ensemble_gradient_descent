import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
#sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')
#sys.path.append('/home/apmd/minseokhwan/gaussian_ensemble_gradient_descent')
#sys.path.append('/ocean/projects/cis260139p/smin2/gaussian_ensemble_gradient_descent/runfiles')
sys.path.append('/home/fs01/sm3266/gaussian_ensemble_gradient_descent')

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import time

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
    precision='float64',
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
    precision='float64',
)


# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
low_fidelity_setting = 24**2 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 46**2 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
t_low_fidelity = 1.13 # low-fidelity simulation time in seconds
t_high_fidelity = 10.30 # high-fidelity simulation time in seconds
t_fwd_AD = 11.89
cost_obj.set_accuracy(high_fidelity_setting)
cost_obj_upsampled.set_accuracy(high_fidelity_setting)
N_convergence = 100

data_dir = "/RCWA_functions/RGB_coupler/run_until_convergence"

t1 = time.time()
print('### Running Simulations', flush=True)

print('\n\t*GEGD', end='', flush=True)
cost_all_GEGD = np.zeros(10)
t_sim_GEGD = np.zeros(10)
convergence_rate_GEGD = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD/RGB_coupler_IPR1_Nensemble20_Ndim60x263_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[-N_convergence:] <= best_cost_hist[-1] + 1e-3):
                t_sim_GEGD[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence])
                cost_all_GEGD[i] = best_cost_hist[j]
                convergence_rate_GEGD[i] = np.abs(best_cost_hist[0] - cost_all_GEGD[i]) / t_sim_GEGD[i]
                break

idx_best = np.argmin(cost_all_GEGD)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD[idx_best], flush=True)

with np.load(directory + data_dir + "/GEGD/RGB_coupler_IPR1_Nensemble20_Ndim60x263_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(idx_best + 1) + "_GEGD_density_hist.npz") as data:
    x = data['best_x_hist'][int(convergence_iter_GEGD[idx_best]),:].reshape(Nx, Ny)

T_tgt, E_in_plane, x_grid, y_grid, z_grid, transmitted_power, incident_power, coeffs = cost_obj.get_diffraction_and_fields(x, 1)
x_up = np.where(gaussian_filter(zoom(x.astype(np.float64), upsampling_ratio, order=1, mode='wrap'), sigma=upsampling_ratio, mode='wrap') > 0.5, 1, 0)
T_tgt_up, E_in_plane_up, x_grid_up, y_grid_up, z_grid_up, transmitted_power_up, incident_power_up, coeffs_up = cost_obj_upsampled.get_diffraction_and_fields(x_up, upsampling_ratio)

print('\n\t*TF-BFGS', end='', flush=True)
cost_all_BFGS = np.zeros(10)
cost_all_BFGS_mfs = np.zeros(10)
t_sim_BFGS = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/TF_BFGS/RGB_coupler_IPR1_Ntrial17_Ndim60x263_D1_mfs7_try" + str(i + 1) + "_TF_results.npz") as data:
        cost_fin = data['cost_fin']
        cost_hist = data['cost_hist']
        cost_all_BFGS[i] = np.min(cost_fin[0,:])
        cost_all_BFGS_mfs[i] = np.min(cost_fin[1,:])
        t_sim_BFGS[i] = t_fwd_AD * (cost_hist.size)

idx_best = np.argmin(cost_all_BFGS)
idx_best_mfs = np.argmin(cost_all_BFGS_mfs)
print(' --> Best Cost (idx=',idx_best+1,', idx_best_mfs=',idx_best_mfs,'): ',cost_all_BFGS[idx_best],' / ',cost_all_BFGS_mfs[idx_best_mfs],' (mfs not enforced / enforced)', flush=True)

print('\n\t*sep-CMA-ES', end='', flush=True)
cost_all_sep_CMA_ES = np.zeros(10)
t_sim_sep_CMA_ES = np.zeros(10)
convergence_rate_sep_CMA_ES = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/sep_CMA_ES/RGB_coupler_IPR1_Nensemble20_Ndim60x263_D1_mfs7_try" + str(i + 1) + "_sep_CMA_ES_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[-N_convergence:] <= best_cost_hist[-1] + 1e-3):
                t_sim_sep_CMA_ES[i] = 20 * t_high_fidelity * (j - N_convergence)
                cost_all_sep_CMA_ES[i] = best_cost_hist[j]
                convergence_rate_sep_CMA_ES[i] = np.abs(best_cost_hist[0] - cost_all_sep_CMA_ES[i]) / t_sim_sep_CMA_ES[i]
                break

idx_best = np.argmin(cost_all_sep_CMA_ES)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_sep_CMA_ES[idx_best], flush=True)

print('\n\t*GA', end='', flush=True)
cost_all_GA = np.zeros(10)
t_sim_GA = np.zeros(10)
convergence_rate_GA = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GA/RGB_coupler_IPR1_Nensemble20_Ndim60x263_D1_mfs7_try" + str(i + 1) + "_AF_GA_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[-N_convergence:] <= best_cost_hist[-1] + 1e-3):
                t_sim_GA[i] = 20 * t_high_fidelity * (j - N_convergence)
                cost_all_GA[i] = best_cost_hist[j]
                convergence_rate_GA[i] = np.abs(best_cost_hist[0] - cost_all_GA[i]) / t_sim_GA[i]
                break

idx_best = np.argmin(cost_all_GA)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GA[idx_best], flush=True)

print('\n\t*AF-STE', end='', flush=True)
cost_all_AF_STE = np.zeros(10)
t_sim_AF_STE = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/AF_STE/RGB_coupler_IPR1_Ntrial17_Ndim60x263_D1_eta0.01_mfs7_try" + str(i + 1) + "_AF_STE_results.npz") as data:
        cost_hist = data['cost_hist']
        converged_iter = np.zeros(cost_hist.shape[1])
        for k in range(cost_hist.shape[1]):
            for j in range(cost_hist.shape[0]):
                if j >= N_convergence and np.argmin(cost_hist[:,k]) <= (j - N_convergence):
                    converged_iter[k] = j - N_convergence
                    t_sim_AF_STE[i] += t_high_fidelity * (j - N_convergence)
                    break
        cost_all_AF_STE[i] = np.min(cost_hist[:np.max(converged_iter)+1,:])

idx_best = np.argmin(cost_all_AF_STE)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_AF_STE[idx_best], flush=True)

np.savez(directory + data_dir + '/RGB_coupler_IPR1_Nensemble20_Ndim60x263_D1_mfs7_simulations',
    cost_all_GEGD=cost_all_GEGD,
    cost_all_BFGS=cost_all_BFGS,
    cost_all_BFGS_mfs=cost_all_BFGS_mfs,
    cost_all_sep_CMA_ES=cost_all_sep_CMA_ES,
    cost_all_GA=cost_all_GA,
    cost_all_AF_STE=cost_all_AF_STE,
    t_sim_GEGD=t_sim_GEGD,
    t_sim_BFGS=t_sim_BFGS,
    t_sim_sep_CMA_ES=t_sim_sep_CMA_ES,
    t_sim_GA=t_sim_GA,
    t_sim_AF_STE=t_sim_AF_STE,
    convergence_rate_GEGD=convergence_rate_GEGD,
    convergence_rate_sep_CMA_ES=convergence_rate_sep_CMA_ES,
    convergence_rate_GA=convergence_rate_GA,
    x=x,
    x_up=x_up,
    T_tgt=T_tgt,
    E_in_plane=E_in_plane,
    x_grid=x_grid,
    y_grid=y_grid,
    z_grid=z_grid,
    transmitted_power=transmitted_power,
    incident_power=incident_power,
    coeffs=coeffs,
    T_tgt_up=T_tgt_up,
    E_in_plane_up=E_in_plane_up,
    x_grid_up=x_grid_up,
    y_grid_up=y_grid_up,
    z_grid_up=z_grid_up,
    transmitted_power_up=transmitted_power_up,
    incident_power_up=incident_power_up,
    coeffs_up=coeffs_up,
)

t2 = time.time()
print('>>> Time taken: ' + str(np.round(t2 - t1, 2)) + ' s', flush=True)