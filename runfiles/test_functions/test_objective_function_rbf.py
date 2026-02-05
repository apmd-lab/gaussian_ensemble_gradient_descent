import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-15])

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import torch
import time
from itertools import product
import gegd.parameter_processing.symmetry_operations as symOp
import gegd.parameter_processing.density_transforms as dtf

class custom_objective:
    def __init__(self, cuda_ind, symmetry, periodic, Nx, Ny, Ndim, min_feature_size, feasible_design_generation_method, n_seed, N_minima, scale, grayscale=False):
                 
        torch.set_warn_always(False)
        self.device = 'cuda:' + str(cuda_ind) if torch.cuda.is_available() else 'cpu'
        self.Nx = Nx
        self.Ny = Ny
        
        # Generate GBest
        np.random.seed(n_seed)
        x = (2*np.random.rand(Ndim, N_minima) - 1)*np.random.rand(N_minima)[np.newaxis,:]
        x = np.hstack((x, -x))
        np.random.seed()
        
        x_gbest = None

        if grayscale:
            for i in range(2*N_minima):
                sigma_filter = min_feature_size / 2
                x_fp = (dtf.filter_and_project(x[:,i], symmetry, periodic, Nx, Ny, min_feature_size, sigma_filter, 8) + 1)/2
                x_gbest_temp = symOp.symmetrize(x_fp, symmetry, Nx, Ny)

                if x_gbest is None:
                    x_gbest = x_gbest_temp.reshape(-1).copy()
                else:
                    x_gbest = np.vstack((x_gbest, x_gbest_temp.reshape(-1)))
            x_gbest = x_gbest.T

        else:
            x_gbest = dtf.binarize(
                x.T,
                symmetry,
                periodic,
                Nx,
                Ny,
                min_feature_size,
                'circle',
                8,
                min_feature_size/2,
                method=feasible_design_generation_method,
                print_runtime_details=True,
            ).T
        
        coeff = scale*np.ones(2*N_minima)
        beta = 2*np.pi*coeff**2
        
        self.x_gbest = torch.tensor(x_gbest, dtype=torch.float64, device=self.device)
        self.coeff = torch.tensor(coeff, dtype=torch.float64, device=self.device)
        self.beta = torch.tensor(beta, dtype=torch.float64, device=self.device)
        
        np.savez(directory[:-15] + '/test_obj_' + str(symmetry) + '_rbf_seed' + str(n_seed) + '_gbest', x_gbest=x_gbest, x=x, coeff=coeff, beta=beta)

    def set_accuracy(self, sigma):
        self.sigma = sigma

    def rbf_cost(self, x):
        dist = torch.sum((self.x_gbest - x[:,None])**2, dim=0)/(self.Nx*self.Ny)
        cost = -torch.sum(self.coeff*torch.exp(-self.beta*dist))

        return cost

    def gaussian_noise(self, x):
        key = abs(hash(x.tobytes())) % (2**63)
    
        rng = np.random.default_rng(key)
        noise = rng.normal()
        
        return noise

    def get_cost(self, x, get_grad=False):
        x = torch.tensor(x, dtype=torch.float64, device=self.device)
        
        if get_grad:
            x.requires_grad_(True)

            cost = self.rbf_cost(x)
            
            cost.backward()
            jac = x.grad.detach().cpu().numpy()
            x.grad = None
        else:
            cost = self.rbf_cost(x)

        cost = cost.detach().cpu().numpy() + self.sigma*self.gaussian_noise(x.detach().cpu().numpy())

        if get_grad:
            return cost, jac.reshape(-1)
        else:
            return cost