import os
directory = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import torch
import optimizer.ensemble_optimization as ENSEMBLE
import optimizer.brush_optimization as CONV
import optimizer.PSO as PSO
from itertools import product
import time
import util.read_mat_data as rmd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Nthreads', type=int, default=1)
parser.add_argument('--brush_size', type=int, default=7)
parser.add_argument('--upsample_ratio', type=int, default=1)
args = parser.parse_args()

Nthreads = args.Nthreads

# Geometry
Nx = 70
Ny = 70
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 0
brush_size = args.brush_size
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

fsp_suffix = 'ensemble_best'

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
low_fidelity_setting = 2 #1 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 5 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
cost_obj.set_accuracy(high_fidelity_setting)

t1 = time.time()
print('### Running Simulations', flush=True)

print('\tEnsemble', flush=True)
suffix = 'mode_converter_Ndim70x70_D1_upsample1_sigma0.01_coeffExp20_brush7_try1.npz'
with np.load(directory + "/results/mode_converter/ensemble_ADAM_results_x_hist_fin_" + suffix) as data:
    x = data['best_x_hist'][-1,:].reshape(Nx*upsample_ratio, Ny*upsample_ratio)
T21, loss, E_fwd, x_field, y_field = cost_obj.get_power_and_fields(x)
np.savez(directory + '/results/ensemble_simulation_' + suffix, T21=T21, loss=loss, E_fwd=E_fwd, x_field=x_field, y_field=y_field)

t2 = time.time()
print('>>> Time taken: ' + str(np.round(t2 - t1, 2)) + ' s', flush=True)