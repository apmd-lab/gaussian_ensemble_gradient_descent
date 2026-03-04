import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')

import numpy as np
import time
import gegd.parameter_processing.density_transforms as dtf

Nthreads = 8

# Geometry
Nx = 60
Ny = 120
symmetry = 0 # Currently supported: (None), (D1,2,4)
periodic = 0
min_feature_size = 7 # minimum feature size in pixels
feasible_design_generation_method = 'brush' # brush / two_phase_projection

if symmetry == 0:
    Ndim = Nx*Ny
   
elif symmetry == 1:
    Ndim = int(np.floor(Nx/2 + 0.5)*Ny)

elif symmetry == 2:
    Ndim = int(np.floor(Nx/2 + 0.5)*np.floor(Ny/2 + 0.5))

elif symmetry == 4:
    Ndim = int(np.floor(Nx/2 + 0.5)*(np.floor(Nx/2 + 0.5) + 1)/2)

# Define Cost Object
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import MDM_demux_ceviche as objfun

IPR_exponent = 1/5

lam_tgt = 1.55

design_dim = np.array([1.75, 3.5])
waveguide_width = 0.5

dpix = design_dim[0]/Nx
waveguide_width_npix = np.round(waveguide_width/dpix)
if waveguide_width_npix < min_feature_size:
    print('\t!!! Waveguide width smaller than the minimum feature size !!!', flush=True)
    assert False
if Ny % 2 == 0:
    waveguide_width_npix += (waveguide_width_npix % 2)*np.sign(waveguide_width/dpix - waveguide_width_npix)
else:
    waveguide_width_npix += (waveguide_width_npix % 2 == 0)*np.sign(waveguide_width/dpix - waveguide_width_npix)
waveguide_halfwidth_npix = int(np.floor(waveguide_width_npix/2 + 0.5))
center_y = int(np.floor(Ny/2 + 0.5))

waveguide_bottom_input = int(center_y - waveguide_halfwidth_npix)
waveguide_top_input = int(waveguide_bottom_input + waveguide_width_npix)
waveguide_bottom_outputTM0 = int(center_y + 2 * waveguide_width_npix - waveguide_halfwidth_npix)
waveguide_top_outputTM0 = int(waveguide_bottom_outputTM0 + waveguide_width_npix)
waveguide_bottom_outputTM1 = int(center_y - waveguide_halfwidth_npix)
waveguide_top_outputTM1 = int(waveguide_bottom_outputTM1 + waveguide_width_npix)
waveguide_bottom_outputTM2 = int(center_y - 2 * waveguide_width_npix - waveguide_halfwidth_npix)
waveguide_top_outputTM2 = int(waveguide_bottom_outputTM2 + waveguide_width_npix)

padding = -np.ones((Nx + 2*min_feature_size, Ny + 2*min_feature_size))
padding[min_feature_size:-min_feature_size,min_feature_size:-min_feature_size] = 0

padding[:min_feature_size,waveguide_bottom_input + min_feature_size:waveguide_top_input + min_feature_size] = 1
padding[-min_feature_size:,waveguide_bottom_outputTM0 + min_feature_size:waveguide_top_outputTM0 + min_feature_size] = 1
padding[-min_feature_size:,waveguide_bottom_outputTM1 + min_feature_size:waveguide_top_outputTM1 + min_feature_size] = 1
padding[-min_feature_size:,waveguide_bottom_outputTM2 + min_feature_size:waveguide_top_outputTM2 + min_feature_size] = 1

mat_padding = 'SiO2_bulk'
mat_waveguide = 'Si_Schinke_Shkondin'

cost_obj = objfun.custom_objective(
    lam_tgt, # in um
    design_dim, # in um
    Nx,
    Ny,
    waveguide_width, # in um
    mat_padding,
    mat_waveguide,
    padding,
    min_feature_size,
    IPR_exponent,
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
    padding=padding,
    method=feasible_design_generation_method,
    Nthreads=Nthreads,
    print_runtime_details=True,
)
t2 = time.time()
brush_time = t2 - t1

upsampling_ratio = np.array([1,1.5,2,2.5,3])
    
cost_all = np.zeros((n_struct, upsampling_ratio.size))
sim_time = np.zeros((n_struct, upsampling_ratio.size))
sim_time_fwd_adj = np.zeros((n_struct, upsampling_ratio.size))

for nb in range(10):
    print('\tStructure ' + str(nb), flush=True)

    for nu in range(upsampling_ratio.size):
        print('\t\tUpsampling Ratio: ' + str(upsampling_ratio[nu]), end='', flush=True)
        cost_obj.set_accuracy(upsampling_ratio[nu])
        x = x_brush_all[nb,:].reshape(Nx, Ny)

        cost = cost_obj.get_cost(x, False)
        time_temp = np.zeros(1)
        for i in range(1):
            t1 = time.time()
            cost = cost_obj.get_cost(x, False)
            t2 = time.time()
            time_temp[i] = t2 - t1
        sim_time[nb,nu] = np.mean(time_temp)
        
        cost_all[nb,nu] = cost
        print(' | Fwd Time: ' + str(sim_time[nb,nu]) + ' s', end='', flush=True)
        
        cost, jac = cost_obj.get_cost(x, True)
        time_temp = np.zeros(1)
        for i in range(1):
            t1 = time.time()
            cost, jac = cost_obj.get_cost(x, True)
            t2 = time.time()
            time_temp[i] = t2 - t1
        sim_time_fwd_adj[nb,nu] = np.mean(time_temp)

        print(' | Fwd+Adj Time: ' + str(sim_time_fwd_adj[nb,nu]) + ' s', flush=True)
        
        np.savez("MDM_demux_convergence_test_ceviche_Nx" + str(Nx) + "_Ny" + str(Ny) + "_mfs" + str(min_feature_size),
            cost_all=cost_all,
            brush_time=brush_time,
            sim_time=sim_time,
            x_brush_all=x_brush_all,
            sim_time_fwd_adj=sim_time_fwd_adj
        )