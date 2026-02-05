import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/apmd/minseokhwan/gaussian_ensemble_gradient_descent')

import numpy as np
import torch
from itertools import product
import time
import gegd.parameter_processing.density_transforms as dtf

Nthreads = 1
cuda_ind = 0

# Geometry
Nx = 45
Ny = 90
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
minimum_feature_size = 7 # minimum feature size in pixels
feasible_design_generation_method = 'two_phase_projection' # brush / two_phase_projection
upsample_ratio = 1

if symmetry == 0:
    Ndim = Nx*Ny
   
elif symmetry == 1:
    Ndim = int(np.floor(Nx*upsample_ratio/2 + 0.5)*Ny*upsample_ratio)

elif symmetry == 2:
    Ndim = int(np.floor(Nx*upsample_ratio/2 + 0.5)*np.floor(Ny*upsample_ratio/2 + 0.5))

elif symmetry == 4:
    Ndim = int(np.floor(Nx*upsample_ratio/2 + 0.5)*(np.floor(Nx*upsample_ratio/2 + 0.5) + 1)/2)

# Define Cost Object
#----------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, save_t_array=False) --> return cost
# (2) set_accuracy(n_harmonic)
#----------------------------------------------------------
import diffraction_grating as objfun

lam = np.array([0.450]) # um
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
                                   minimax=False,
                                   cuda_ind=cuda_ind)

cost_obj.set_geometry(Nx*upsample_ratio, Ny*upsample_ratio, period, thickness)
cost_obj.set_source(lam=lam, angle_inc=angle_inc)

# Convergence Test
print('### Convergence Test')

n_struct = 10
np.random.seed(100)
x = 2*np.random.rand(n_struct, Ndim) - 1
t1 = time.time()
x_brush_all = dtf.binarize(
    x,
    symmetry,
    periodic,
    Nx,
    Ny,
    minimum_feature_size,
    'circle',
    8,
    minimum_feature_size*np.sqrt(2)/4,
    method=feasible_design_generation_method,
    print_runtime_details=True,
)
t2 = time.time()
brush_time = t2 - t1

max_harmonic = 22
cost_all = np.zeros((n_struct, int(max_harmonic/2)))
sim_time = np.zeros((n_struct, int(max_harmonic/2)))
sim_time_AD = np.zeros((n_struct, int(max_harmonic/2)))

for nb in range(n_struct):
    print('\tStructure ' + str(nb), flush=True)
    
    for nh in range(2, max_harmonic, 2):
        print('\t\tHarmonics: ' + str(nh), end='', flush=True)
        cost_obj.set_accuracy([int(nh/2),nh])
        x = x_brush_all[nb,:].reshape(Nx*upsample_ratio, Ny*upsample_ratio)
    
        t1 = time.time()
        cost = cost_obj.get_cost(x, False)
        t2 = time.time()
        sim_time[nb,int(nh/2)-1] = t2 - t1
        
        cost_all[nb,int(nh/2)-1] = cost
        print(' | Fwd Time: ' + str(sim_time[nb,int(nh/2)-1]) + ' s', end='', flush=True)
        
        t1 = time.time()
        cost = cost_obj.get_cost(x, True)
        t2 = time.time()
        sim_time_AD[nb,int(nh/2)-1] = t2 - t1
        
        print(' | Fwd+AD Time: ' + str(sim_time_AD[nb,int(nh/2)-1]) + ' s', flush=True)
        
        np.savez("diffraction_grating_convergence_test_Nx" + str(Nx) + "_Ny" + str(Ny) + "_mfs" + str(minimum_feature_size),
            cost_all=cost_all,
            brush_time=brush_time,
            sim_time=sim_time,
            x_brush_all=x_brush_all,
            sim_time_AD=sim_time_AD,
        )