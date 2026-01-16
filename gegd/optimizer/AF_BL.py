import numpy as np
import torch
from torch.func import jacrev
import gegd.parameter_processing.bilevel_design_generator as bdg
import time
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD

class optimizer:
    def __init__(
		self,
		Nx,
		Ny,
		Ntrial,
		symmetry,
		periodic,
		padding,
		maxiter,
		high_fidelity_setting,
		brush_size,
		upsample_ratio=1,
		beta_proj=8,
		cost_obj=None,
		Nthreads=1,
	):
        
        self.Nx = Nx
        self.Ny = Ny
        self.Ntrial = Ntrial
        self.symmetry = symmetry
        self.periodic = periodic
        self.padding = padding
        self.maxiter = maxiter
        self.beta_proj = beta_proj
        self.beta_proj_sigma = beta_proj
        self.brush_size = brush_size
        self.upsample_ratio = upsample_ratio
        self.sigma_filter = brush_size/2/np.sqrt(2)
        self.high_fidelity_setting = high_fidelity_setting
        self.cost_obj = cost_obj
        self.Nthreads = Nthreads
        
        # Get Number of Independent Parameters
        if symmetry == 0:
            self.Ndim = Nx*Ny
           
        elif symmetry == 1:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*Ny)
        
        elif symmetry == 2:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*np.floor(Ny/2 + 0.5))
        
        elif symmetry == 4:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*(np.floor(Nx/2 + 0.5) + 1)/2)
	
    def bilevel_cost(self, x_reward):
        # Get Bilevel Binarized Densities ------------------------------------------------------------
        generator = bdg.conditional_generator(
			self.Nx,
			self.Ny,
			self.symmetry,
			self.periodic,
			self.padding,
			self.brush_size + 2,
		)

        x_bin = np.zeros((self.Ntrial, self.Nx*self.Ny))
        f = np.zeros(self.Ntrial)
        jac = np.zeros((self.Ntrial, self.Ndim))

        for n in range(self.Ntrial):
            x_reward_temp = torch.tensor(x_reward[n,:], device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float64)
            jac_near_binary, x_near_binary = jacrev(generator.generate_near_binary, has_aux=True)(x_reward_temp)
            x_bin[n,:] = np.where(x_near_binary.detach().cpu().numpy() < 0.5, 0, 1).reshape(-1)

            # Sample Cost Function --------------------------------------------------------------
            t1 = time.time()
            f[n], jac_temp = self.cost_obj.get_cost(x_bin[n,:], get_grad=True)
            t2 = time.time()
            #print(t2 - t1, flush=True)

            jac[n,:] = jac_temp.T @ jac_near_binary.detach().cpu().numpy()
        
        return f, jac, x_bin

    def ADAM(self,
             x,
             beta_ADAM1,
             beta_ADAM2,
             eta_ADAM,
             jac_mean=None,
             jac_var=None,
             adam_iter=None,
             ):
        
        if jac_mean is None:
            jac_mean = np.zeros_like(x)
        if jac_var is None:
            jac_var = np.zeros_like(x)
        if adam_iter is None:
            adam_iter = 0
            
        while True:
            t1 = time.time()
            adam_iter += 1
            
            loss, jac, x_bin = self.bilevel_cost(x)

            best_sample = np.argmin(loss)

            if self.best_x_latent_hist is None:
                self.best_x_latent_hist = x[best_sample,:].copy()
            else:
                self.best_x_latent_hist = np.vstack((self.best_x_latent_hist, x[best_sample,:]))
                
            if self.best_x_hist is None:
                self.best_x_hist = x_bin[best_sample,:].copy()
            else:
                self.best_x_hist = np.vstack((self.best_x_hist, x_bin[best_sample,:]))
            
            if self.cost_hist is None:
                self.cost_hist = loss.copy()
            else:
                self.cost_hist = np.vstack((self.cost_hist, loss))
        
            t2 = time.time()
            t_rem = (t2 - t1)*(self.maxiter - self.n_iter + 1)/3600
            
            if comm.rank == 0:
                print('    | %4d |  %8.3f |   %5.2f   |' %(
                    self.n_iter,
                    np.min(loss),
                    t_rem,
                    ),
                    end='',
					flush=True,
				)

            self.save_data(
				x=x,
				jac_mean=jac_mean,
				jac_var=jac_var,
				adam_iter=adam_iter,
			)

            if adam_iter >= self.maxiter:
                break
                
            # Update Average Gradients
            jac_mean = beta_ADAM1*jac_mean + (1 - beta_ADAM1)*jac
            jac_var = beta_ADAM2*jac_var + (1 - beta_ADAM2)*jac**2
            
            # Unbias Average Gradients
            jac_mean_unbiased = jac_mean/(1 - beta_ADAM1**adam_iter)
            jac_var_unbiased = jac_var/(1 - beta_ADAM2**adam_iter)
            
            x -= eta_ADAM*jac_mean_unbiased/(np.sqrt(jac_var_unbiased) + 1e-8)
            
            self.n_iter += 1
		
    def run(self, n_seed, output_filename, x0=None, eta=0.01, load_data=False):
        if comm.rank == 0:
            print('### Bilevel Optimization (seed = ' + str(n_seed) + ')\n', flush=True)
    
        np.random.seed(n_seed)
        self.output_filename = output_filename
        
        if load_data:
            data_file1 = output_filename + "_AF_BL_results.npz"
            data_file2 = output_filename + "_AF_BL_density_hist.npz"
                
            with np.load(data_file1) as data:
                self.n_iter = data['n_iter']
                self.best_cost_hist = data['best_cost_hist'][:self.n_iter]

                adam_iter = data['adam_iter'] - 1
                
            with np.load(data_file2) as data:
                self.best_x_latent_hist = data['best_x_latent_hist'][:self.n_iter,:]
                self.best_x_hist = data['best_x_hist'][:self.n_iter,:]
                
                x0 = data['x_bounded']
                jac_mean = data['jac_mean']
                jac_var = data['jac_var']
            
        else:
            self.n_iter = 0
            
            self.best_x_latent_hist = None
            self.best_x_hist = None
            self.best_cost_hist = np.zeros(0)
            
            if x0 is None:
                # Initial Structure
                x0 = np.random.uniform(-1, 1, size=(self.Ntrial, self.Ndim))
            
            jac_mean = None
            jac_var = None
            adam_iter = None
        
        if comm.rank == 0:
            print('    | Iter | Cost Best | t_rem(hr) |', flush=True)

        eta_ADAM = eta*np.ones(self.Ndim)
        self.ADAM(
            x0,
			0.9,
			0.999,
			eta_ADAM,
            jac_mean=jac_mean,
			jac_var=jac_var,
			adam_iter=adam_iter,
		)
        
        self.save_data()
    
    def save_data(self, x_bounded=None, jac_mean=None, jac_var=None, adam_iter=None):
        if comm.rank == 0:
            if x_bounded is None:
                with np.load(self.output_filename + "_AF_BL_results.npz") as data:
                    adam_iter = data['adam_iter']
                
                with np.load(self.output_filename + "_AF_BL_density_hist.npz") as data:
                    x_bounded = data['x_bounded']
                    jac_mean = data['jac_mean']
                    jac_var = data['jac_var']
        
            # Customize below
            np.savez(self.output_filename + "_AF_BL_results",
                best_cost_hist=self.best_cost_hist,
                n_iter=self.n_iter,
                adam_iter=adam_iter,
            )
						
            np.savez(self.output_filename + "_AF_BL_density_hist",
                best_x_latent_hist=self.best_x_latent_hist,
                best_x_hist=self.best_x_hist,
                x_bounded=x_bounded,
                jac_mean=jac_mean,
                jac_var=jac_var,
			)