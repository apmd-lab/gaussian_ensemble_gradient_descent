import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')

import numpy as np
import jax
import time
import gegd.parameter_processing.density_transforms as dtf

Nthreads = 16

# Geometry
Nx = 100
Ny = 100
symmetry = 3 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
min_feature_size = 7 # minimum feature size in pixels
d_pixel = 0.02 # pixel side length (nm)
feasible_design_generation_method = 'brush' # brush / two_phase_projection

if symmetry == 0:
    Ndim = Nx*Ny
   
elif symmetry == 1:
    Ndim = int(np.floor(Nx/2 + 0.5)*Ny)

elif symmetry == 2:
    Ndim = int(np.floor(Nx/2 + 0.5)*np.floor(Ny/2 + 0.5))

elif symmetry == 3:
    Ndim = int(np.floor(Nx + 0.5)*(np.floor(Nx + 0.5) + 1)/2)

elif symmetry == 4:
    Ndim = int(np.floor(Nx/2 + 0.5)*(np.floor(Nx/2 + 0.5) + 1)/2)

# Define Cost Object
#----------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, save_t_array=False) --> return cost
# (2) set_accuracy(n_harmonic)
#----------------------------------------------------------
import RGB_color_router_FMMAX as objfun

lam = np.array([0.650,0.550,0.450]) # um
theta_inc = np.array([0])*np.pi/180
phi_inc = np.array([0])*np.pi/180
in_plane_wavevector = np.array([0.0, 0.0])

period = np.array([Nx * d_pixel, Ny * d_pixel])
thickness_background = 1.0
thickness_capping_layer = 0.0
thickness_pattern = 0.7
thickness_spacer = 1.0
thickness_substrate = 1.0

mat_background = np.array(['Air']) # background (incident side)
mat_capping_layer = np.array(['Air'])
mat_pattern = np.array(['Air','Si3N4_Luke']) # Low RI, High RI
mat_spacer = np.array(['Air'])
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
    IPR_exponent=1/1,
)

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
    min_feature_size,
    'circle',
    8.0,
    min_feature_size/2,
    method=feasible_design_generation_method,
    Nthreads=Nthreads,
    print_runtime_details=True,
)
t2 = time.time()
brush_time = t2 - t1

load_data = False

n_harmonic = np.arange(2, 46)**2

if load_data:
    with np.load("RGB_color_router_convergence_test_FMMAX_JONES_DIRECT_Nx" + str(Nx) + "_Ny" + str(Ny) + "_mfs" + str(min_feature_size) + ".npz") as data:
        cost_all_temp = data['cost_all']
        sim_time_temp = data['sim_time']
        sim_time_AD_temp = data['sim_time_AD']
        n_start = 0 #np.argmin(sim_time[:,0,0]) - 1
        nh_start = 25

    cost_all = np.zeros((n_struct, n_harmonic.size))
    sim_time = np.zeros((n_struct, n_harmonic.size))
    sim_time_AD = np.zeros((n_struct, n_harmonic.size))

    cost_all[:,:nh_start] = cost_all_temp
    sim_time[:,:nh_start] = sim_time_temp
    sim_time_AD[:,:nh_start] = sim_time_AD_temp

else:
    cost_all = np.zeros((n_struct, n_harmonic.size))
    sim_time = np.zeros((n_struct, n_harmonic.size))
    sim_time_AD = np.zeros((n_struct, n_harmonic.size))
    n_start = 0
    nh_start = 0

for nb in range(n_start, n_struct):
    print('\tStructure ' + str(nb), flush=True)
    
    for nh in range(nh_start, n_harmonic.size):
        print('\t\tHarmonics: ' + str(n_harmonic[nh]), end='', flush=True)
        cost_obj.set_accuracy(n_harmonic[nh])
        x = x_brush_all[nb,:].reshape(Nx, Ny)
    
        cost = cost_obj.get_cost(x, False) # account for JIT compilation
        time_temp = np.zeros(1)
        for i in range(1):
            t1 = time.time()
            cost = cost_obj.get_cost(x, False)
            t2 = time.time()
            time_temp[i] = t2 - t1
        jax.clear_caches()
        sim_time[nb,nh] = np.mean(time_temp)
    
        cost_all[nb,nh] = cost
        print(' | Fwd Time: ' + str(sim_time[nb,nh]) + ' s', end='', flush=True)
    
        cost = cost_obj.get_cost(x, True)
        time_temp = np.zeros(1)
        for i in range(1):
            t1 = time.time()
            cost = cost_obj.get_cost(x, True)
            t2 = time.time()
            time_temp[i] = t2 - t1
        jax.clear_caches()
        sim_time_AD[nb,nh] = np.mean(time_temp)
    
        print(' | Fwd+AD Time: ' + str(sim_time_AD[nb,nh]) + ' s', flush=True)
    
        np.savez("RGB_color_router_convergence_test_FMMAX_JONES_DIRECT_Nx" + str(Nx) + "_Ny" + str(Ny) + "_mfs" + str(min_feature_size),# + "_10reps",
            cost_all=cost_all,
            brush_time=brush_time,
            sim_time=sim_time,
            x_brush_all=x_brush_all,
            sim_time_AD=sim_time_AD,
        )