import os
directory = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import torch
from itertools import product
import optimizer.error_robustness as ER
import time
import util.read_mat_data as rmd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Nthreads', type=int, default=1)
parser.add_argument('--upsample_ratio', type=int, default=1)
args = parser.parse_args()

Nthreads = args.Nthreads

# Geometry
Nx = 35
Ny = 70
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
upsample_ratio = args.upsample_ratio

# Define Cost Object
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import TORCWA_functions.diffraction_grating as objfun

lam = np.array([0.633]) # um
theta_inc = np.array([0])*np.pi/180
phi_inc = np.array([0])*np.pi/180
angle_inc = np.array(list(product(theta_inc, phi_inc)))

theta_tgt = 45*np.pi/180
max_diff_order = np.array([0,1])
Ly = max_diff_order[1]*np.max(lam)/np.sin(theta_tgt)
period = np.array([Ly*Nx/Ny,Ly])
thickness = 0.3

mat_multilayer = np.array(['Si_Schinke_Shkondin']) # Top to bottom
mat_background = np.array(['SiO2_bulk','Air']) # background (incident side), background (exit side)

cost_obj = objfun.custom_objective(mat_background,
                                   mat_multilayer,
                                   Nthreads,
                                   minimax=False)#(args.optimizer=='ensemble' and upsample_ratio > 1))

cost_obj.set_geometry(Nx*upsample_ratio, Ny*upsample_ratio, period, thickness)
cost_obj.set_source(lam=lam, angle_inc=angle_inc)

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
low_fidelity_setting = [4,8] # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = [10,20] # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
cost_obj.set_accuracy(high_fidelity_setting)

perturbation_brush_size = 3
Nsample = 100

t1 = time.time()
print('### Running Analysis', flush=True)

print('\tGrayscale', flush=True)
suffix = 'diffraction_grating_Ndim35x70_D1_upsample1_sigma0.01_coeffExp5_brush7_try5'
with np.load(directory + "/results/diffraction_grating/grayscale_results_fin_" + suffix + '.npz') as data:
    x = data['x_fin'][-1,:].reshape(Nx, Ny)
ER.evaluate_error_robustness(x, cost_obj, perturbation_brush_size, Nsample, periodic, 'grayscale_' + suffix)

#print('\tBrush', flush=True)
#suffix = 'diffraction_grating_Ndim35x70_D1_upsample1_sigma0.01_coeffExp5_brush7_try6'
#with np.load(directory + "/results/brush_results_x_hist_fin_" + suffix + '.npz') as data:
#    x = data['x_hist'][484,:].reshape(Nx, Ny)
#ER.evaluate_error_robustness(x, cost_obj, perturbation_brush_size, Nsample, periodic, 'brush_' + suffix)

print('\tEnsemble', flush=True)
suffix = 'diffraction_grating_Ndim35x70_D1_upsample1_sigma0.01_coeffExp20_brush7_try3'
with np.load(directory + "/results/diffraction_grating/ensemble_ADAM_results_x_hist_fin_" + suffix + '.npz') as data:
    x = data['best_x_hist'][-1,:].reshape(Nx, Ny)
ER.evaluate_error_robustness(x, cost_obj, perturbation_brush_size, Nsample, periodic, 'ensemble_' + suffix)

#print('\tPSO', flush=True)
#suffix = 'diffraction_grating_Ndim35x70_D1_upsample1_sigma0.01_coeffExp5_brush7_try3'
#with np.load(directory + "/results/PSO_results_x_hist_fin_" + suffix + '.npz') as data:
#    x = data['gbest_x_binary_hist'][-1,:].reshape(Nx, Ny)
#ER.evaluate_error_robustness(x, cost_obj, perturbation_brush_size, Nsample, periodic, 'grayscale_' + suffix)

t2 = time.time()
print('>>> Time taken: ' + str(np.round(t2 - t1, 2)) + ' s', flush=True)