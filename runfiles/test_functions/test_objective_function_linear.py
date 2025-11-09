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
import optimizer.symmetry_operations as symOp
import optimizer.density_transforms as dtf

class custom_objective:
    def __init__(self, cuda_ind, symmetry, periodic, Nx, Ny, Ndim, brush_size, brush_shape, n_seed):
                 
        torch.set_warn_always(False)
        self.device = 'cuda:' + str(cuda_ind) if torch.cuda.is_available() else 'cpu'
        self.Nx = Nx
        self.Ny = Ny
        
        # Generate GBest
        np.random.seed(n_seed)
        x = 2*np.random.rand(Ndim) - 1
        x_noise = 2*np.random.rand(Ndim) - 1
        np.random.seed()

        x_brush = dtf.binarize(x, symmetry, periodic, Nx, Ny, brush_size, brush_shape, 8, brush_size*np.sqrt(2)/4)
        x_noise = dtf.binarize(x_noise, symmetry, periodic, Nx, Ny, brush_size, brush_shape, 8, brush_size*np.sqrt(2)/4)
        
        np.savez(directory[:-15] + '/results/test_obj_linear_seed' + str(n_seed) + '_gbest', x_gbest=x_brush, x_noise=x_noise)
        
        self.x_gbest = torch.tensor(x_brush, dtype=torch.float64, device=self.device)
        self.x_noise = torch.tensor(x_noise, dtype=torch.float64, device=self.device)

    def set_accuracy(self, coeff):
        self.coeff = coeff

    def linear_cost(self, x):
        weights = 2*self.x_gbest - 1
        cost = -torch.mean(weights*(2*x - 1))
        weights_noise = 2*self.x_noise - 1
        cost_noise = -torch.mean(weights_noise*(2*x - 1))

        return (1 - self.coeff)*cost + self.coeff*cost_noise

    def get_cost(self, x, get_grad=False):
        x = torch.tensor(x, dtype=torch.float64, device=self.device)
        
        if get_grad:
            x.requires_grad_(True)

            cost = self.linear_cost(x)
            
            cost.backward()
            jac = x.grad.detach().cpu().numpy()
            x.grad = None
        else:
            cost = self.linear_cost(x)

        if get_grad:
            return cost.detach().cpu().numpy(), jac.reshape(-1)
        else:
            return cost.detach().cpu().numpy()