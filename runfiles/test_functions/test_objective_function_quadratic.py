import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-15])

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import time
from itertools import product
import util.read_mat_data as rmd
import optimizer.brush_generator as brush

class custom_objective:
    def __init__(self, cuda_ind, Nx, Ny, Ndim, brush_size, brush_shape, n_seed):
                 
        torch.set_warn_always(False)
        self.device = 'cuda:' + str(cuda_ind) if torch.cuda.is_available() else 'cpu'
        self.Nx = Nx
        self.Ny = Ny
        
        # Generate GBest
        np.random.seed(n_seed)
        x = 2*np.random.rand(Ndim) - 1
        np.random.seed()

        if Nx % 2 == 0:
            x_quart = np.zeros((int(Nx/2), int(Ny/2)))
            x_quart[np.triu_indices(int(Nx/2))] = x
            x_quart = x_quart + x_quart.T - np.diag(np.diag(x_quart))
            
            x_sym = np.zeros((Nx, Ny))
            x_sym[:int(Nx/2),:int(Ny/2)] = x_quart
            x_sym[int(Nx/2):,:int(Ny/2)] = np.rot90(x_quart)
            x_sym[int(Nx/2):,int(Ny/2):] = np.rot90(x_quart, k=2)
            x_sym[:int(Nx/2),int(Ny/2):] = np.rot90(x_quart, k=-1)
        
        else:
            x_quart = np.zeros((int((Nx+1)/2), int((Ny+1)/2)))
            x_quart[np.triu_indices(int((Nx+1)/2))] = x
            x_quart = x_quart + x_quart.T - np.diag(np.diag(x_quart))
            
            x_sym = np.zeros((Nx, Ny))
            x_sym[:int((Nx+1)/2),:int((Ny+1)/2)] = x_quart
            x_sym[int((Nx+1)/2):,:int((Ny+1)/2)] = np.rot90(x_quart)[1:,:]
            x_sym[int((Nx+1)/2):,int((Ny+1)/2):] = np.rot90(x_quart, k=2)[1:,1:]
            x_sym[:int((Nx+1)/2),int((Ny+1)/2):] = np.rot90(x_quart, k=-1)[:,1:]
        
        if brush_size is not None:
            x_filter = gaussian_filter(x_sym, sigma=brush_size/2, mode='wrap')
        else:
            x_filter = x_sym.copy()
        
        x_proj = np.tanh(8*x_filter)
        
        if brush_size is not None:
            x_brush = brush.make_feasible(x_proj, Nx, Ny, brush_size, brush_shape, padding=None, periodic=True, symmetry='D4', dim=2)
        
        np.savez(directory[:-15] + '/results/test_obj_linear_seed' + str(n_seed) + '_gbest', x_gbest=x_brush)
        
        self.x_gbest = torch.tensor(x_brush, dtype=torch.float64, device=self.device)

    def set_accuracy(self, dummy):
        pass

    def quadratic_cost(self, x):
        cost = 2*torch.mean((self.x_gbest - x)**2) - 1
        cost_exp = -torch.exp(-5*cost)
        
        return cost_exp

    def get_cost(self, x, get_grad=False):
        x = torch.tensor(x, dtype=torch.float64, device=self.device)
        
        if get_grad:
            x.requires_grad_(True)

            cost = self.quadratic_cost(x)
            
            cost.backward()
            jac = x.grad.detach().cpu().numpy()
            x.grad = None
        else:
            cost = self.quadratic_cost(x)

        if get_grad:
            return cost.detach().cpu().numpy(), jac.reshape(-1)
        else:
            return cost.detach().cpu().numpy()