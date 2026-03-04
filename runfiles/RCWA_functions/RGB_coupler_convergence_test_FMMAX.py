import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')

import numpy as np
import jax
import time
import gegd.parameter_processing.density_transforms as dtf

Nthreads = 8

# Geometry
Nx = 60
Ny = 263
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
min_feature_size = 7 # minimum feature size in pixels
d_pixel = 0.01 # pixel side length (nm)
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
import RGB_coupler_FMMAX as objfun

incident_pol = 'TM' # TE / TM

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
    IPR_exponent=1/1,
    incident_pol=incident_pol,
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

n_harmonic = np.arange(6, 40)**2

if load_data:
    with np.load("RGB_coupler_convergence_test_FMMAX_JONES_DIRECT_" + incident_pol + "_Nx" + str(Nx) + "_Ny" + str(Ny) + "_mfs" + str(min_feature_size) + ".npz") as data:
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
    
        np.savez("RGB_coupler_convergence_test_FMMAX_JONES_DIRECT_" + incident_pol + "_Nx" + str(Nx) + "_Ny" + str(Ny) + "_mfs" + str(min_feature_size),# + "_10reps",
            cost_all=cost_all,
            brush_time=brush_time,
            sim_time=sim_time,
            x_brush_all=x_brush_all,
            sim_time_AD=sim_time_AD,
        )