import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import time

Nthreads = 8

# Geometry
Nx = 100
Ny = 100
symmetry = 3 # Currently supported: (None), (D1,2,4)
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
import RCWA_functions.RGB_color_router_FMMAX as objfun

IPR_exponent = 1/1

lam = np.array([0.650,0.550,0.450]) # um
theta_inc = np.array([0])*np.pi/180
phi_inc = np.array([0])*np.pi/180
in_plane_wavevector = np.array([0.0, 0.0])

period = np.array([Nx * d_pixel, Ny * d_pixel])
thickness_background = 1.0
thickness_capping_layer = 0.0
thickness_pattern = 0.7
thickness_spacer = 2.0
thickness_substrate = 1.0

mat_background = np.array(['Air']) # background (incident side)
mat_capping_layer = np.array(['SiO2_bulk'])
mat_pattern = np.array(['SiO2_bulk','Si3N4_Luke']) # Low RI, High RI
mat_spacer = np.array(['SiO2_bulk'])
mat_substrate = np.array(['Si_Schinke_Shkondin'])

cost_obj = objfun.custom_objective(
    Nx,
    Ny,
    period,
    thickness_background,
    thickness_capping_layer,
    thickness_pattern,
    thickness_spacer,
    thickness_substrate,
    lam,
    in_plane_wavevector,
    mat_background,
    mat_capping_layer,
    mat_pattern,
    mat_spacer,
    mat_substrate,
    IPR_exponent=IPR_exponent,
)

cost_obj_upsampled = objfun.custom_objective(
    Nx * upsampling_ratio,
    Ny * upsampling_ratio,
    period,
    thickness_background,
    thickness_capping_layer,
    thickness_pattern,
    thickness_spacer,
    thickness_substrate,
    lam,
    in_plane_wavevector,
    mat_background,
    mat_capping_layer,
    mat_pattern,
    mat_spacer,
    mat_substrate,
    IPR_exponent=IPR_exponent,
)

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
low_fidelity_setting = 14**2 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 24**2 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
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
with np.load(directory + "/RGB_color_router_IPR1_Nensemble10_Ndim100x100_D3_sig_ens0.01_eta2e-05_mfs7_exp20_try1_GEGD_results.npz") as data:
    x = data['best_x_final'].reshape(Nx, Ny)
flux, R_flux, G_flux, B_flux, incident_flux = cost_obj.get_detector_flux(x)
x_up = np.where(gaussian_filter(zoom(x.astype(np.float64), upsampling_ratio, order=1, mode='wrap'), sigma=upsampling_ratio, mode='wrap') > 0.5, 1, 0)
flux_up, R_flux_up, G_flux_up, B_flux_up, incident_flux_up = cost_obj_upsampled.get_detector_flux(x_up)
np.savez(directory + '/RGB_color_router_IPR1_Nensemble10_Ndim100x100_D3_sig_ens0.01_eta2e-05_mfs7_exp20_try1_GEGD_simulation',
    flux=flux,
    R_flux=R_flux,
    G_flux=G_flux,
    B_flux=B_flux,
    incident_flux=incident_flux,
    x=x,
    x_up=x_up,
    flux_up=flux_up,
    R_flux_up=R_flux_up,
    G_flux_up=G_flux_up,
    B_flux_up=B_flux_up,
    incident_flux_up=incident_flux_up,
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