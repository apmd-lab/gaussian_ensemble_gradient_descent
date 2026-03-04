import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')

import numpy as np
import time
import gegd.parameter_processing.density_transforms as dtf

Nthreads = 48

# Geometry
Nx = 100
Ny = 100
symmetry = 0 # Currently supported: (None), (D1,2,4)
periodic = 0
min_feature_size = 7 # minimum feature size in pixels
feasible_design_generation_method = 'brush' # brush / two_phase_projection
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
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import PRS_lumFDTD as objfun

IPR_exponent = 1/3

c = 299792458
um = 1e-6
nm = 1e-9

lam_tgt = 1.55*um

design_dim = np.array([2.5*um, 2.5*um, 400*nm])
waveguide_width = 0.5*um

dpix = design_dim[0]/Nx
waveguide_width_npix = np.round(waveguide_width/dpix)
if waveguide_width_npix < min_feature_size:
    print('\t!!! Waveguide width smaller than the minimum feature size !!!', flush=True)
    assert False
if Nx % 2 == 0:
    waveguide_width_npix += (waveguide_width_npix % 2)*np.sign(waveguide_width/dpix - waveguide_width_npix)
else:
    waveguide_width_npix += (waveguide_width_npix % 2 == 0)*np.sign(waveguide_width/dpix - waveguide_width_npix)
waveguide_halfwidth_npix = int(np.floor(waveguide_width_npix/2 + 0.5))
center_x = int(np.floor(Nx/2 + 0.5))

waveguide_bottom_input = int(center_x + waveguide_width_npix - waveguide_halfwidth_npix)
waveguide_top_input = int(waveguide_bottom_input + waveguide_width_npix)
waveguide_bottom_outputTE0 = int(center_x + waveguide_width_npix - waveguide_halfwidth_npix)
waveguide_top_outputTE0 = int(waveguide_bottom_outputTE0 + waveguide_width_npix)
waveguide_bottom_outputTM0 = int(center_x - waveguide_width_npix - waveguide_halfwidth_npix)
waveguide_top_outputTM0 = int(waveguide_bottom_outputTM0 + waveguide_width_npix)

padding = -np.ones((Nx + 2*min_feature_size, Ny + 2*min_feature_size))
padding[min_feature_size:-min_feature_size,min_feature_size:-min_feature_size] = 0

padding[waveguide_bottom_input + min_feature_size:waveguide_top_input + min_feature_size,:min_feature_size] = 1
padding[waveguide_bottom_outputTE0 + min_feature_size:waveguide_top_outputTE0 + min_feature_size,-min_feature_size:] = 1
padding[waveguide_bottom_outputTM0 + min_feature_size:waveguide_top_outputTM0 + min_feature_size,-min_feature_size:] = 1

mat_padding = 'SiO2_bulk'
mat_waveguide = 'Si_Schinke_Shkondin'

fsp_suffix = 'convtest1'

cost_obj = objfun.custom_objective(2,
                                   lam_tgt, # in m
                                   design_dim, # in m
                                   waveguide_width, # in m
                                   Nx,
                                   Ny,
                                   mat_padding,
                                   mat_waveguide,
                                   IPR_exponent,
                                   Nthreads,
                                   fsp_suffix,
                                   symmetric_bounds=False)

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
    print_runtime_details=True,
)
t2 = time.time()
brush_time = t2 - t1

upsampling_ratio_fdtd = np.array([1/2,1,2,3,4,5,6])
    
cost_all = np.zeros((n_struct, upsampling_ratio_fdtd.size))
sim_time = np.zeros((n_struct, upsampling_ratio_fdtd.size))
sim_time_fwd_adj = np.zeros((n_struct, upsampling_ratio_fdtd.size))

for nb in range(10):
    print('\tStructure ' + str(nb), flush=True)

    for nu in range(upsampling_ratio_fdtd.size):
        print('\t\tUpsampling Ratio: ' + str(upsampling_ratio_fdtd[nu]), end='', flush=True)
        cost_obj.set_accuracy(upsampling_ratio_fdtd[nu])
        x = x_brush_all[nb,:].reshape(Nx*upsample_ratio, Ny*upsample_ratio)
        #x = (x_all[nb,brush_size:-brush_size,brush_size:-brush_size].copy() + 1)/2*(1 - 2e-3) + 1e-3

        t1 = time.time()
        cost = cost_obj.get_cost(x, False)
        t2 = time.time()
        sim_time[nb,nu] = t2 - t1
        
        cost_all[nb,nu] = cost
        print(' | Fwd Time: ' + str(sim_time[nb,nu]) + ' s', end='', flush=True)
        
        t1 = time.time()
        cost, jac = cost_obj.get_cost(x, True)
#        np.savez(directory + '/debug_adjoint_jac0', jac=jac)
#        assert False
#        jac_FD = np.zeros((Nx, Ny))
#        for nx in range(15, Nx):
#            for ny in range(60):
#                print('Nx: ' + str(nx) + ' | Ny: ' + str(ny), flush=True)
#                x1 = x.copy().astype(np.float64)
#                x1[nx,ny] -= 1e-3
#                f1 = cost_obj.get_cost(x1, False)
#                
#                x2 = x.copy().astype(np.float64)
#                x2[nx,ny] += 1e-3
#                f2 = cost_obj.get_cost(x2, False)
#                
#                jac_FD[nx,ny] = (f2 - f1)/2e-3
#        np.savez(directory + '/debug_adjoint_jac', jac=jac, jac_FD=jac_FD)
#        assert False
        t2 = time.time()
        sim_time_fwd_adj[nb,nu] = t2 - t1

        print(' | Fwd+Adj Time: ' + str(sim_time_fwd_adj[nb,nu]) + ' s', flush=True)
        
        np.savez("PRS_convergence_test_Nx" + str(Nx) + "_Ny" + str(Ny) + "_mfs" + str(min_feature_size),
            cost_all=cost_all,
            brush_time=brush_time,
            sim_time=sim_time,
            x_brush_all=x_brush_all,
            sim_time_fwd_adj=sim_time_fwd_adj
        )