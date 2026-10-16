import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
#sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')
#sys.path.append('/home/apmd/minseokhwan/gaussian_ensemble_gradient_descent')
#sys.path.append('/u/smin2/gaussian_ensemble_gradient_descent')
sys.path.append('/home/x-smin2/gaussian_ensemble_gradient_descent')

Nthreads = 128

import numpy as np
import time
import gegd.parameter_processing.density_transforms as dtf

# Geometry
Nx = 70
Ny = 70
symmetry = 0 # Currently supported: (None), (D1,2,4)
periodic = 0
min_feature_size = 7 # minimum feature size in pixels
upsample_ratio = 1
feasible_design_generation_method = 'brush'

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
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import FDTD_functions.mode_converter_ceviche as objfun

lam_tgt = 1.55

design_dim = np.array([1*lam_tgt,1*lam_tgt])
waveguide_width = 0.5 # in um

dpix = design_dim[1]/Ny
waveguide_width_npix = np.round(waveguide_width/dpix)
if waveguide_width_npix < brush_size:
    print('\t!!! Waveguide width smaller than the minimum feature size !!!', flush=True)
    assert False
if Ny % 2 == 0:
    waveguide_width_npix += (waveguide_width_npix % 2)*np.sign(waveguide_width/dpix - waveguide_width_npix)
else:
    waveguide_width_npix += (waveguide_width_npix % 2 == 0)*np.sign(waveguide_width/dpix - waveguide_width_npix)
waveguide_halfwidth_npix = int(np.floor(waveguide_width_npix/2 + 0.5))
center_y = int(np.floor(Ny/2 + 0.5))
waveguide_bottom = int(center_y - waveguide_halfwidth_npix)
waveguide_top = int(waveguide_bottom + waveguide_width_npix)

padding = -np.ones((Nx + 2*brush_size, Ny + 2*brush_size))
padding[brush_size:-brush_size,brush_size:-brush_size] = 0
padding[:brush_size,waveguide_bottom + brush_size:waveguide_top + brush_size] = 1
padding[-brush_size:,waveguide_bottom + brush_size:waveguide_top + brush_size] = 1

mat_padding = 'SiO2_bulk'
mat_waveguide = 'Si_Schinke_Shkondin'

cost_obj = objfun.custom_objective(
    lam_tgt, # in um
    design_dim, # in um
    Nx*upsample_ratio,
    Ny*upsample_ratio,
    waveguide_width, # in um
    mat_padding,
    mat_waveguide,
)

# Convergence Test
print('### Convergence Test')

n_struct = 100
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

if comm.rank == 0:
    np.savez("/mode_converter_convergence_test_brush" + str(brush_size) + "_Nx" + str(Nx) + "_Ny" + str(Ny),
        brush_time=brush_time,
        x_brush_all=x_brush_all,
    )

upsampling_ratio_fdtd = np.linspace(0.2, 5.0, 25)

#with np.load(directory[:-15] + "/results/integrated_bandpass_filter_convergence_test_brush" + str(brush_size) + ".npz") as data:
#    cost_all = data['cost_all']
#    sim_time = data['sim_time']
#    sim_time_fwd_adj = data['sim_time_fwd_adj']
    
cost_all = np.zeros((n_struct, upsampling_ratio_fdtd.size))
sim_time = np.zeros((n_struct, upsampling_ratio_fdtd.size))
sim_time_fwd_adj = np.zeros((n_struct, upsampling_ratio_fdtd.size))

for nb in range(n_struct):
    print('\tStructure ' + str(nb), flush=True)

    for nu in range(upsampling_ratio_fdtd.size):
        print('\t\tUpsampling Ratio: ' + str(upsampling_ratio_fdtd[nu]), end='', flush=True)
        cost_obj.set_accuracy(upsampling_ratio_fdtd[nu])
        x = x_brush_all[nb,:].reshape(Nx*upsample_ratio, Ny*upsample_ratio)

        t1 = time.time()
        cost = cost_obj.get_cost(x, False)
        t2 = time.time()
        sim_time[nb,nu] = t2 - t1
        
        cost_all[nb,nu] = cost
        print(' | Fwd Time: ' + str(sim_time[nb,nu]) + ' s', end='', flush=True)
        
        t1 = time.time()
        cost, jac = cost_obj.get_cost(x, True)
        t2 = time.time()
        sim_time_fwd_adj[nb,nu] = t2 - t1

        print(' | Fwd+Adj Time: ' + str(sim_time_fwd_adj[nb,nu]) + ' s', flush=True)
        
        np.savez("mode_converter_convergence_test_ceviche_Nx" + str(Nx) + "_Ny" + str(Ny) + "_mfs" + str(min_feature_size),
            cost_all=cost_all,
            brush_time=brush_time,
            sim_time=sim_time,
            x_brush_all=x_brush_all,
            sim_time_fwd_adj=sim_time_fwd_adj
        )