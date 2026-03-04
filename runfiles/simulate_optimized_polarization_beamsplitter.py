import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import time

Nthreads = 8

# Geometry
Nx = 45
Ny = 90
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
import RCWA_functions.polarization_beamsplitter_FMMAX as objfun

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
    IPR_exponent=1/1,
)

cost_obj_upsampled = objfun.custom_objective(
    Nx * upsampling_ratio,
    Ny * upsampling_ratio,
    period,
    thickness,
    lam,
    in_plane_wavevector,
    mat_background,
    mat_pattern,
    diff_order,
    IPR_exponent=1/1,
)

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
low_fidelity_setting = 19**2 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 36**2 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
cost_obj.set_accuracy(high_fidelity_setting)
cost_obj_upsampled.set_accuracy(high_fidelity_setting)

t1 = time.time()
print('### Running Simulations', flush=True)

#print('\tGrayscale', flush=True)
#suffix = 'diffraction_grating_Ndim35x70_D1_sigma0.01_brush7_try10.npz'
#with np.load(directory + "/results/grayscale_results_x_hist_fin_" + suffix) as data:
#    x = data['x_hist'][-2,:].reshape(Nx, Ny)
#Txx, Tyy, y_field, z_field, ReExx, ReEyy = cost_obj.get_transmission_and_fields(x)
#np.savez(directory + '/results/grayscale_simulation_' + suffix, Txx=Txx, Tyy=Tyy, y_field=y_field, z_field=z_field, ReExx=ReExx, ReEyy=ReEyy, x=x)
#
#print('\tBrush', flush=True)
#suffix = 'diffraction_grating_Ndim35x70_D1_upsample1_sigma0.01_coeffExp5_brush7_try6.npz'
#with np.load(directory + "/results/brush_results_x_hist_fin_" + suffix) as data:
#    x = data['x_hist'][484,:].reshape(Nx, Ny)
#Txx, Tyy, y_field, z_field, ReExx, ReEyy = cost_obj.get_transmission_and_fields(x)
#np.savez(directory + '/results/brush_simulation_' + suffix, Txx=Txx, Tyy=Tyy, y_field=y_field, z_field=z_field, ReExx=ReExx, ReEyy=ReEyy, x=x)

print('\tGEGD', flush=True)
with np.load(directory + "/polarization_beamsplitter_IPR1_Nensemble10_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try1_GEGD_results.npz") as data:
    x = data['best_x_final'].reshape(Nx, Ny)
T_TE, T_TM, Ex, Ey, x_grid, y_grid, z_grid, transmitted_power, incident_power, coeffs, idx_tgt = cost_obj.get_diffraction_and_fields(x)
x_up = np.where(gaussian_filter(zoom(x.astype(np.float64), upsampling_ratio, order=1, mode='wrap'), sigma=upsampling_ratio, mode='wrap') > 0.5, 1, 0)
T_TE_up, T_TM_up, Ex_up, Ey_up, x_grid_up, y_grid_up, z_grid_up, transmitted_power_up, incident_power_up, coeffs_up, idx_tgt_up = cost_obj_upsampled.get_diffraction_and_fields(x_up)
np.savez(directory + '/polarization_beamsplitter_IPR1_Nensemble10_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try1_GEGD_simulation',
    x=x,
    x_up=x_up,
    T_TE=T_TE,
    T_TM=T_TM,
    Ex=Ex,
    Ey=Ey,
    x_grid=x_grid,
    y_grid=y_grid,
    z_grid=z_grid,
    transmitted_power=transmitted_power,
    incident_power=incident_power,
    coeffs=coeffs,
    idx_tgt=idx_tgt,
    T_TE_up=T_TE_up,
    T_TM_up=T_TM_up,
    Ex_up=Ex_up,
    Ey_up=Ey_up,
    x_grid_up=x_grid_up,
    y_grid_up=y_grid_up,
    z_grid_up=z_grid_up,
    transmitted_power_up=transmitted_power_up,
    incident_power_up=incident_power_up,
    coeffs_up=coeffs_up,
    idx_tgt_up=idx_tgt_up,
)

#print('\tPSO', flush=True)
#suffix = 'diffraction_grating_Ndim35x70_D1_upsample1_sigma0.01_coeffExp5_brush7_try3.npz'
#with np.load(directory + "/results/PSO_results_x_hist_fin_" + suffix) as data:
#    x = data['gbest_x_binary_hist'][-1,:].reshape(Nx, Ny)
#Txx, Tyy, y_field, z_field, ReExx, ReEyy = cost_obj.get_transmission_and_fields(x)
#np.savez(directory + '/results/PSO_simulation_' + suffix, Txx=Txx, Tyy=Tyy, y_field=y_field, z_field=z_field, ReExx=ReExx, ReEyy=ReEyy, x=x)
'''
print('\tsep-CMA-ES', flush=True)
with np.load(directory + "/RCWA_functions/sep_CMA_ES/RGB_coupler_IPR5_Nensemble20_Ndim37x269_D1_mfs9_try1_sep_CMA_ES_results.npz") as data:
    x = data['best_x_final'].reshape(Nx, Ny)
Txx, y_field, z_field, ReExx = cost_obj.get_diffraction_and_fields(x)
np.savez(directory + '/RCWA_functions/sep_CMA_ES/RGB_coupler_IPR5_Nensemble20_Ndim37x269_D1_mfs9_try1_sep_CMA_ES_simulation',
    Txx=Txx, y_field=y_field, z_field=z_field, ReExx=ReExx, x=x)
'''
t2 = time.time()
print('>>> Time taken: ' + str(np.round(t2 - t1, 2)) + ' s', flush=True)