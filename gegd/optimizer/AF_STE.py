import numpy as np
import gegd.parameter_processing.density_transforms as dtf
import gegd.parameter_processing.symmetry_operations as symOp
import time

from mpi4py import MPI
comm = MPI.COMM_WORLD

class optimizer:
    def __init__(self,
                 Nx,
                 Ny,
                 Ntrial,
                 symmetry,
                 periodic,
                 padding,
                 high_fidelity_setting,
                 brush_size,
                 upsample_ratio=1,
                 beta_proj=8,
                 brush_shape='circle',
                 cost_obj=None,
                 Nthreads=1,
                 ):
                       
        self.Nx = Nx
        self.Ny = Ny
        self.Ntrial = Ntrial
        self.symmetry = symmetry
        self.periodic = periodic
        self.padding = padding
        self.beta_proj = beta_proj
        self.brush_size = brush_size
        self.brush_shape = brush_shape
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
    
    def straight_through_jacobian(self, x0):
        # Get Brush Binarized Densities ------------------------------------------------------------
        x_bin = dtf.binarize(x0, self.symmetry, self.periodic, self.Nx, self.Ny, self.brush_size, self.brush_shape, self.beta_proj, self.sigma_filter, padding=self.padding, Nthreads=self.Nthreads)

        # Sample Modified Cost Function --------------------------------------------------------------
        self.cost_obj.set_accuracy(self.high_fidelity_setting)

        f0 = np.zeros(self.Ntrial)
        jac_STE = np.zeros((self.Ntrial, self.Ndim))
        for n in range(self.Ntrial):
            f0[n], jac_temp = self.cost_obj.get_cost(x_bin[n,:], get_grad=True)
            jac_sym = jac_temp.reshape(self.Nx, self.Ny)
            jac_STE[n,:] = dtf.backprop_filter_and_project(jac_sym, x0[n,:], self.symmetry, self.periodic, self.Nx, self.Ny, self.brush_size, self.sigma_filter, self.beta_proj, padding=self.padding)
        
        return f0, jac_STE, x_bin
    
    def ADAM(self,
             x_bounded,
             lb,
             ub,
             beta_ADAM1,
             beta_ADAM2,
             eta_ADAM,
             maxiter,
             beta=0,
             jac_mean=None,
             jac_var=None,
             adam_iter=None,
             ):
    
        # Dummy Variables
        x = -np.log((ub[np.newaxis,:] - lb[np.newaxis,:])/(x_bounded - lb[np.newaxis,:]) - 1)
        x[:,ub==lb] = ub[ub==lb][np.newaxis,:]
        
        if jac_mean is None:
            jac_mean = np.zeros_like(x)
        if jac_var is None:
            jac_var = np.zeros_like(x)
        if adam_iter is None:
            adam_iter = 0
            
        while True:
            t1 = time.time()
            adam_iter += 1
            
            x_bounded = lb[np.newaxis,:] + (ub[np.newaxis,:] - lb[np.newaxis,:])/(1 + np.exp(-x))

            loss, jac, x_bin = self.straight_through_jacobian(x_bounded)

            jac[:,ub==lb] = 0
            jac *= np.exp(-x)*(ub[np.newaxis,:] - lb[np.newaxis,:])/(1 + np.exp(-x))**2

            if self.cost_hist is None:
                self.cost_hist = loss.copy()
            else:
                self.cost_hist = np.vstack((self.cost_hist, loss))
            
            if self.x_latent_hist is None:
                self.x_latent_hist = x_bounded[np.argmin(loss),:].copy()
            else:
                self.x_latent_hist = np.vstack((self.x_latent_hist, x_bounded[np.argmin(loss),:]))
            
            if self.x_hist is None:
                self.x_hist = x_bin[np.argmin(loss),:].copy()
            else:
                self.x_hist = np.vstack((self.x_hist, x_bin[np.argmin(loss),:]))

            t2 = time.time()

            if comm.rank == 0:
                t_rem = (t2 - t1)*(maxiter - self.n_iter + 1)/3600
                print('    | %12d | %12.5f |   %5.2f   |' %(self.n_iter, np.min(loss), t_rem), flush=True)

            self.save_data(x_bounded=x_bounded,
                           x_bin=x_bin,
                           jac_mean=jac_mean,
                           jac_var=jac_var,
                           adam_iter=adam_iter)

            if adam_iter >= maxiter:
                break

            # Update Average Gradients
            jac_mean = beta_ADAM1*jac_mean + (1 - beta_ADAM1)*jac
            jac_var = beta_ADAM2*jac_var + (1 - beta_ADAM2)*jac**2
            
            # Unbias Average Gradients
            jac_mean_unbiased = jac_mean/(1 - beta_ADAM1**adam_iter)
            jac_var_unbiased = jac_var/(1 - beta_ADAM2**adam_iter)
            
            # Update Variables
            x -= eta_ADAM*jac_mean_unbiased/(np.sqrt(jac_var_unbiased) + 1e-8)
            # eta *= 0.95

            self.n_iter += 1

        return x_bounded
    
    def run(self, n_seed, output_filename, maxiter, eta_ADAM=0.1, load_data=False):
        if comm.rank == 0:
            print('### Brush Optimization (seed = ' + str(n_seed) + ')\n', flush=True)
    
        self.output_filename = output_filename
        self.sigma_filter = self.brush_size/2/np.sqrt(2)

        lb = -np.ones(self.Ndim)
        ub = np.ones(self.Ndim)

        if load_data:
            data_file1 = output_filename + "_AF_STE_results.npz"
            data_file2 = output_filename + "_AF_STE_density_hist.npz"
                
            with np.load(data_file1) as data:
                self.n_iter = data['n_iter']
                self.cost_hist = data['cost_hist'][:self.n_iter,:]
                adam_iter = data['adam_iter'] - 1
                
            with np.load(data_file2) as data:
                self.x_latent_hist = data['x_latent_hist'][:self.n_iter,:]
                self.x_hist = data['x_hist'][:self.n_iter,:]
                x0 = data['x_bounded']
                jac_mean = data['jac_mean']
                jac_var = data['jac_var']
            
        else:
            self.x_latent_hist = None
            self.x_hist = None
            self.cost_hist = None
            
            self.n_iter = 0
        
            # Initial Structure
            if n_seed is not None:
                np.random.seed(n_seed)
            
            x0 = 2*np.random.rand(self.Ntrial, self.Ndim) - 1
            np.random.seed()
            
            jac_mean = None
            jac_var = None
            adam_iter = None

        if comm.rank == 0:
            print('    |  Iteration   |  Best Cost   | t_rem(hr) |', flush=True)
        
        x_bin = self.ADAM(x0, lb, ub, 0.667, 0.9, eta_ADAM, maxiter, jac_mean=jac_mean, jac_var=jac_var, adam_iter=adam_iter)
        
        if comm.rank == 0:
            print('', flush=True)
    
        self.save_data(x_bin=x_bin)
    
    def save_data(self, x_bounded=0, x_bin=0, jac_mean=0, jac_var=0, adam_iter=0):
        if comm.rank == 0:
            np.savez(self.output_filename + "_AF_STE_results",
                     cost_hist=self.cost_hist,
                     n_iter=self.n_iter,
                     adam_iter=adam_iter)
                     
            np.savez(self.output_filename + "_AF_STE_density_hist",
                     x_hist=self.x_hist,
                     x_latent_hist=self.x_latent_hist,
                     x_bounded=x_bounded,
                     x_bin=x_bin,
                     jac_mean=jac_mean,
                     jac_var=jac_var)