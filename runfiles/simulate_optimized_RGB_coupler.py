import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/apmd/minseokhwan/gaussian_ensemble_gradient_descent')

import numpy as np
import torch
from gegd.optimizer import TF_BFGS, GEGD, AF_PSO, AF_GA, sep_CMA_ES, AF_BL
from itertools import product
import time
import util.read_mat_data as rmd

Nthreads = 1
cuda_ind = 1

# Geometry
Nx = 37
Ny = 269
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
min_feature_size = 9 # minimum feature size in pixels
d_pixel = 0.008 # pixel side length (nm)
feasible_design_generation_method = 'two_phase_projection' # brush / two_phase_projection
upsample_ratio = 1

# Define Cost Object
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import RCWA_functions.RGB_coupler as objfun

IPR_exponent = 1/5

lam = np.array([0.675,0.540,0.450]) # um
theta_inc = np.array([0])*np.pi/180
phi_inc = np.array([0])*np.pi/180
angle_inc = np.array(list(product(theta_inc, phi_inc)))

theta_tgt = 45*np.pi/180
diff_order = np.array([
    [0,4],
    [0,5],
    [0,6],
])
period = np.array([Nx * d_pixel, Ny * d_pixel])
thickness = 0.7

mat_pattern = np.array(['Air','TiO2_Sarkar']) # Low RI, High RI
mat_background = np.array(['Air','SiO2_bulk']) # background (incident side), background (exit side)

cost_obj = objfun.custom_objective(mat_background,
                                   mat_pattern,
                                   Nthreads,
                                   diff_order,
                                   IPR_exponent=IPR_exponent,
                                   cuda_ind=cuda_ind)

cost_obj.set_geometry(Nx*upsample_ratio, Ny*upsample_ratio, period, thickness)
cost_obj.set_source(lam=lam, angle_inc=angle_inc)

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
low_fidelity_setting = [5,14] # [5,14], [5,15] # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = [8,20] # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
cost_obj.set_accuracy(high_fidelity_setting)

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
'''
print('\tGEGD', flush=True)
with np.load(directory + "/RCWA_functions/GEGD/RGB_coupler_Nensemble20_Ndim37x269_D1_sig_RBF1.125_sig_ens1e-06_beta_proj16.0_eta1e-09_mfs9_try1_GEGD_results.npz") as data:
    x = data['best_x_final'].reshape(Nx, Ny)
Txx, y_field, z_field, ReExx = cost_obj.get_diffraction_and_fields(x)
np.savez(directory + '/RCWA_functions/GEGD/RGB_coupler_Nensemble20_Ndim37x269_D1_sig_RBF1.125_sig_ens1e-06_beta_proj16.0_eta1e-09_mfs9_try1_GEGD_simulation',
    Txx=Txx, y_field=y_field, z_field=z_field, ReExx=ReExx, x=x)
'''

#print('\tPSO', flush=True)
#suffix = 'diffraction_grating_Ndim35x70_D1_upsample1_sigma0.01_coeffExp5_brush7_try3.npz'
#with np.load(directory + "/results/PSO_results_x_hist_fin_" + suffix) as data:
#    x = data['gbest_x_binary_hist'][-1,:].reshape(Nx, Ny)
#Txx, Tyy, y_field, z_field, ReExx, ReEyy = cost_obj.get_transmission_and_fields(x)
#np.savez(directory + '/results/PSO_simulation_' + suffix, Txx=Txx, Tyy=Tyy, y_field=y_field, z_field=z_field, ReExx=ReExx, ReEyy=ReEyy, x=x)

print('\tsep-CMA-ES', flush=True)
with np.load(directory + "/RCWA_functions/sep_CMA_ES/RGB_coupler_IPR5_Nensemble20_Ndim37x269_D1_mfs9_try1_sep_CMA_ES_results.npz") as data:
    x = data['best_x_final'].reshape(Nx, Ny)
Txx, y_field, z_field, ReExx = cost_obj.get_diffraction_and_fields(x)
np.savez(directory + '/RCWA_functions/sep_CMA_ES/RGB_coupler_IPR5_Nensemble20_Ndim37x269_D1_mfs9_try1_sep_CMA_ES_simulation',
    Txx=Txx, y_field=y_field, z_field=z_field, ReExx=ReExx, x=x)

t2 = time.time()
print('>>> Time taken: ' + str(np.round(t2 - t1, 2)) + ' s', flush=True)