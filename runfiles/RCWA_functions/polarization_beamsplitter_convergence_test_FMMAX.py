import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
#sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')
#sys.path.append('/home/apmd/minseokhwan/gaussian_ensemble_gradient_descent')
sys.path.append('/home/fs01/sm3266/gaussian_ensemble_gradient_descent')

Nthreads = 1
cuda_ind = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_ind)
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
import jax
import time
import gegd.parameter_processing.density_transforms as dtf

# Geometry
Nx = 45
Ny = 90
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
import polarization_beamsplitter_FMMAX as objfun

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
    precision='float64'
)

# Convergence Test
print('### Convergence Test')

n_struct = 7
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

with np.load('polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_mfs7_try4_AF_GA_results.npz') as data:
    x_brush_all[0,:] = data['best_x_binary_final']
with np.load('polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_mfs7_try4_AF_GA_density_hist.npz') as data:
    x_brush_all[1:3,:] = data['best_x_binary_hist'][[183,255],:]
with np.load('polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_mfs7_try4_sep_CMA_ES_results.npz') as data:
    x_brush_all[3,:] = data['best_x_final']
with np.load('polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_mfs7_try4_sep_CMA_ES_density_hist.npz') as data:
    x_brush_all[4:7,:] = data['best_x_hist'][[369,374,375],:]

n_harmonic = np.arange(2, 51)**2

cost_all = np.zeros((n_struct, n_harmonic.size))
sim_time = np.zeros((n_struct, n_harmonic.size))
sim_time_AD = np.zeros((n_struct, n_harmonic.size))

for nh in range(n_harmonic.size):
    print('\tHarmonics: ' + str(n_harmonic[nh]), flush=True)
    cost_obj.set_accuracy(n_harmonic[nh])
    
    # account for JIT compilation
    x_dummy = x_brush_all[0,:].reshape(Nx, Ny)
    _ = cost_obj.get_cost(x_dummy, False)
    _ = cost_obj.get_cost(x_dummy, True)
    
    step = max(1, n_struct // 100)
    for idx, nb in enumerate(range(n_struct)):
        if idx % step == 0 or idx == n_struct - 1:
            pct = (idx + 1) / n_struct
            bar = '█' * int(30 * pct) + '░' * (30 - int(30 * pct))
            print(f'\r\t\t[{bar}] {100*pct:5.1f}% ({idx+1}/{n_struct})', end='', flush=True)
    
        x = x_brush_all[nb,:].reshape(Nx, Ny)
    
        time_temp = np.zeros(1)
        for i in range(1):
            t1 = time.time()
            cost = cost_obj.get_cost(x, False)
            t2 = time.time()
            time_temp[i] = t2 - t1
        sim_time[nb,nh] = np.mean(time_temp)
        cost_all[nb,nh] = cost
        
        time_temp = np.zeros(1)
        for i in range(1):
            t1 = time.time()
            cost = cost_obj.get_cost(x, True)
            t2 = time.time()
            time_temp[i] = t2 - t1
        sim_time_AD[nb,nh] = np.mean(time_temp)
        
    print(f' | Fwd Time: {np.mean(sim_time[:,nh]):.4f} s | Fwd+AD Time: {np.mean(sim_time_AD[:,nh]):.4f} s', flush=True)
    
    jax.clear_caches()

    np.savez("polarization_beamsplitter_convergence_test_FMMAX_JONES_DIRECT_Nx" + str(Nx) + "_Ny" + str(Ny) + "_mfs" + str(min_feature_size),# + "_10reps",
        cost_all=cost_all,
        brush_time=brush_time,
        sim_time=sim_time,
        x_brush_all=x_brush_all,
        sim_time_AD=sim_time_AD,
    )