import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-15])

import numpy as np
from scipy.ndimage import gaussian_filter
import optimizer.brush_generator as brush
from itertools import product
import time
import util.read_mat_data as rmd
import optimizer.symmetry_operations as symOp
import optimizer.density_transforms as dtf

from mpi4py import MPI
comm = MPI.COMM_WORLD

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Nthreads', type=int, default=1)
args = parser.parse_args()

Nthreads = args.Nthreads

if not os.path.exists(directory[:-15] + "/results"):
    os.mkdir(directory[:-15] + "/results")

# Geometry
Nx = 100 #100
Ny = 200 #200
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 0
brush_size = 7 # minimum feature size in pixels
upsample_ratio = 1

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
import FDTD_functions.integrated_bandpass_filter_MEEP as objfun

lam_min = 1.5
lam_max = 1.6
N_freq = 11

design_dim = np.array([2*lam_max,4*lam_max]) #2*lam_max,4*lam_max
waveguide_width = 0.5 # in um

dpix = design_dim[1]/Ny
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

bandpass_min = 1.54
bandpass_max = 1.56

cost_obj = objfun.custom_objective(lam_min, # in um
                                   lam_max,
                                   design_dim, # in um
                                   Nx,
                                   Ny,
                                   N_freq,
                                   waveguide_width, # in um
                                   mat_padding,
                                   mat_waveguide,
                                   bandpass_min,
                                   bandpass_max)

# Convergence Test
print('### Convergence Test')

n_struct = 10
np.random.seed(100)
x = 2*np.random.rand(n_struct, Ndim) - 1
t1 = time.time()
x_all, x_brush_all = dtf.binarize(x, symmetry, periodic, Nx, Ny, brush_size, 'circle', 8, brush_size*np.sqrt(2)/4, upsample_ratio=upsample_ratio, padding=padding, output_details=True, Nthreads=Nthreads)
t2 = time.time()
brush_time = t2 - t1

if comm.rank == 0:
    np.savez(directory[:-15] + "/results/test_integrated_bandpass_filter_convergence_test_brush" + str(brush_size) + "_Nx" + str(Nx) + "_Ny" + str(Ny),
        brush_time=brush_time,
        x_brush_all=x_brush_all,
    )

upsampling_ratio_fdtd = np.array([1,2,3,4,5])

#with np.load(directory[:-15] + "/results/integrated_bandpass_filter_convergence_test_brush" + str(brush_size) + ".npz") as data:
#    cost_all = data['cost_all']
#    sim_time = data['sim_time']
#    sim_time_fwd_adj = data['sim_time_fwd_adj']
    
cost_all = np.zeros((n_struct, upsampling_ratio_fdtd.size))
sim_time = np.zeros((n_struct, upsampling_ratio_fdtd.size))
sim_time_fwd_adj = np.zeros((n_struct, upsampling_ratio_fdtd.size))

for nb in range(n_struct):
    if comm.rank == 0:
        print('\tStructure ' + str(nb), flush=True)

    for nu in range(upsampling_ratio_fdtd.size):
        if comm.rank == 0:
            print('\t\tUpsampling Ratio: ' + str(upsampling_ratio_fdtd[nu]), end='', flush=True)
        cost_obj.set_accuracy(upsampling_ratio_fdtd[nu])
        x = x_brush_all[nb,:].reshape(Nx*upsample_ratio, Ny*upsample_ratio)

        t1 = time.time()
        cost = cost_obj.get_cost(x, False)
        t2 = time.time()
        sim_time[nb,nu] = t2 - t1
        
        cost_all[nb,nu] = cost
        if comm.rank == 0:
            print(' | Fwd Time: ' + str(sim_time[nb,nu]) + ' s', end='', flush=True)
        
        t1 = time.time()
        cost, jac = cost_obj.get_cost(x, True)
#        np.savez(directory + '/debug_adjoint_jac0', jac=jac)
#        assert False
#        jac_FD = np.zeros((Nx, Ny))
#        for nx in range(Nx):
#            for ny in range(Ny):
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

        if comm.rank == 0:
            print(' | Fwd+Adj Time: ' + str(sim_time_fwd_adj[nb,nu]) + ' s', flush=True)
        
        if comm.rank == 0:
            np.savez(directory[:-15] + "/results/test_integrated_bandpass_filter_convergence_test_brush" + str(brush_size) + "_Nx" + str(Nx) + "_Ny" + str(Ny),
                cost_all=cost_all,
                brush_time=brush_time,
                sim_time=sim_time,
                x_all=x_all,
                x_brush_all=x_brush_all,
                sim_time_fwd_adj=sim_time_fwd_adj
            )