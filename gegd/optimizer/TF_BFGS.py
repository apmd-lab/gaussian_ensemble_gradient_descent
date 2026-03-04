import numpy as np
from scipy.optimize import minimize, Bounds
import gegd.parameter_processing.density_transforms as dtf
import gegd.parameter_processing.symmetry_operations as symOp
import gegd.parameter_processing.feasible_design_generator.fdg as FDG
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
                 min_feature_size,
                 upsample_ratio=1,
                 brush_shape='circle',
                 cost_obj=None,
                 Nthreads=1,
                 cuda_ind=0,
                 ):
                       
        self.Nx = Nx
        self.Ny = Ny
        self.Ntrial = Ntrial
        self.symmetry = symmetry
        self.periodic = periodic
        self.padding = padding
        self.min_feature_size = min_feature_size
        self.brush_shape = brush_shape
        self.high_fidelity_setting = high_fidelity_setting
        self.cost_obj = cost_obj
        self.Nthreads = Nthreads
        self.cuda_ind = cuda_ind
        
        self.cost_obj.set_accuracy(self.high_fidelity_setting)
        
        # Get Number of Independent Parameters
        if symmetry == 0:
            self.Ndim = Nx*Ny
           
        elif symmetry == 1:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*Ny)
        
        elif symmetry == 2:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*np.floor(Ny/2 + 0.5))
        
        elif symmetry == 3:
            self.Ndim = int(np.floor(Nx + 0.5)*(np.floor(Nx + 0.5) + 1)/2)
        
        elif symmetry == 4:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*(np.floor(Nx/2 + 0.5) + 1)/2)
    
    def grayscale_cost(self, x0, skip_filter_and_project=False):
        # Get Brush Binarized Densities ------------------------------------------------------------
        if skip_filter_and_project:
            x_fp = x0.copy()
        else:
            x_temp = dtf.filter_and_project(x0, self.symmetry, self.periodic, self.Nx, self.Ny, self.min_feature_size, self.sigma_filter, self.beta, padding=self.padding)
            x_fp = (symOp.symmetrize(x_temp, self.symmetry, self.Nx, self.Ny).reshape(-1) + 1)/2

        # Sample Modified Cost Function --------------------------------------------------------------
        t1 = time.time()
        f0 = self.cost_obj.get_cost(x_fp, get_grad=False)
        t2 = time.time()
        #print(t2 - t1, flush=True)
        
        if self.cost_hist is None:
            self.cost_hist = np.array([f0])
        else:
            self.cost_hist = np.append(self.cost_hist, f0)
        
        if self.x_latent_hist is None:
            self.x_latent_hist = x0.copy()
        else:
            if skip_filter_and_project:
                self.x_latent_hist = np.vstack((self.x_latent_hist, self.x_latent_hist[-1,:]))
            else:
                self.x_latent_hist = np.vstack((self.x_latent_hist, x0))
        
        if self.x_hist is None:
            self.x_hist = x_fp.copy()
        else:
            self.x_hist = np.vstack((self.x_hist, x_fp))

        if comm.rank == 0:
            t_rem = 2*(t2 - t1)*(self.Ntrial*self.maxiter - self.n_iter + 1)/3600
            print('    | %12d |  %6.2f  | %12.5f |   %5.2f   |' %(self.n_iter, self.beta, f0, t_rem), flush=True)

        self.save_data(x0=x0,
                       beta=self.beta)
        
        self.n_iter += 1
        
        return f0
    
    def grayscale_jacobian(self, x0):
        # Get Brush Binarized Densities ------------------------------------------------------------
        x_temp = dtf.filter_and_project(x0, self.symmetry, self.periodic, self.Nx, self.Ny, self.min_feature_size, self.sigma_filter, self.beta, padding=self.padding)
        x_fp = (symOp.symmetrize(x_temp, self.symmetry, self.Nx, self.Ny).reshape(-1) + 1)/2

        # Sample Modified Cost Function --------------------------------------------------------------
        t1 = time.time()
        _, jac_temp = self.cost_obj.get_cost(x_fp, get_grad=True)
            
        jac_sym = jac_temp.reshape(self.Nx, self.Ny)
        jac = dtf.backprop_filter_and_project(jac_sym, x0, self.symmetry, self.periodic, self.Nx, self.Ny, self.min_feature_size, self.sigma_filter, self.beta, padding=self.padding)
        t2 = time.time()
        
        return jac
    
    def BFGS(self, x0, maxiter_beta):
        bounds = Bounds(lb=-1, ub=1, keep_feasible=True)
        results = minimize(
            self.grayscale_cost,
            x0,
            method='L-BFGS-B',
            jac=self.grayscale_jacobian,
            bounds=bounds,
            options={'maxfun': maxiter_beta},
        )

        return results.x
    
    def run(self, n_seed, output_filename, maxiter, beta_init=8.0, beta_ratio=2.0, n_beta=5, load_data=False):
        if comm.rank == 0:
            print('### Grayscale Optimization (seed = ' + str(n_seed) + ')\n', flush=True)
    
        self.output_filename = output_filename
        self.maxiter = maxiter
        self.n_beta = n_beta
        self.sigma_filter = self.min_feature_size / 2

        if load_data:
            data_file1 = output_filename + "_TF_results.npz"
            #data_file2 = output_filename + "_TF_density_hist.npz"
                
            with np.load(data_file1) as data:
                i_start = 0
                n_start = data['ntrial']
                self.n_iter = data['n_iter']
                self.cost_hist = data['cost_hist']
                self.cost_fin = data['cost_fin']
                self.x_fin = data['x_fin']
                
            #with np.load(data_file2) as data:
            #    self.x_latent_hist = data['x_latent_hist']
            #    self.x_hist = data['x_hist']
            self.x_latent_hist = None
            self.x_hist = None
            
        else:
            self.x_latent_hist = None
            self.x_hist = None
            self.cost_hist = None
            self.cost_fin = np.zeros((2, self.Ntrial))
            self.x_fin = np.zeros((2, self.Ntrial, self.Nx*self.Ny))
            
            i_start = 0
            n_start = 0
            self.n_iter = 0
        
        # Initial Structure
        if n_seed is not None:
            np.random.seed(n_seed)
        
        x0 = 2*np.random.rand(self.Ntrial, self.Ndim) - 1
        np.random.seed()
        
        self.beta = beta_init
        self.maxiter_beta = (self.maxiter/self.n_beta)*np.ones(self.n_beta)

        for n in range(n_start, self.Ntrial):
            self.ntrial = n
            
            if comm.rank == 0:
                print('### Trial ' + str(n) + '\n', flush=True)
                print('    |  Iteration   |   beta   |  Best Cost   | t_rem(hr) |', flush=True)
        
            x0_n = x0[n,:].copy()
            for i in range(i_start, self.n_beta):
                x0_n = self.BFGS(x0_n, self.maxiter_beta[i])
                self.beta *= beta_ratio
            
            self.beta = np.inf
            f0 = self.grayscale_cost(x0_n)
            self.cost_fin[0,n] = f0
            self.x_fin[0,n,:] = self.x_hist[-1,:].copy()

            x_reward = 2 * self.x_hist[-1,:].reshape(1, self.Nx, self.Ny) - 1
            x0_feasible = FDG.make_feasible_parallel(
                x_reward.astype(np.float32),
                self.min_feature_size,
                self.periodic,
                self.symmetry,
                2,
                1,
                self.Nthreads,
            ).reshape(-1)
            f0_feasible = self.grayscale_cost(x0_feasible, skip_filter_and_project=True)
            self.cost_fin[1,n] = f0_feasible
            self.x_fin[1,n,:] = self.x_hist[-1,:].copy()

            self.beta = beta_init
            
            self.save_data()
        
            if comm.rank == 0:
                print('', flush=True)
    
    def save_data(self, x0=0, beta=0):
        if comm.rank == 0:
            if self.x_hist.ndim == 1:
                x_hist_final = self.x_hist.copy()
            else:
                x_hist_final = self.x_hist[-1,:].copy()
            
            np.savez(self.output_filename + "_TF_results",
                     cost_hist=self.cost_hist,
                     n_iter=self.n_iter,
                     ntrial=self.ntrial,
                     cost_fin=self.cost_fin,
                     x_fin=self.x_fin,
                     beta=beta,
                     x_hist_final=x_hist_final)
                     
            #np.savez(self.output_filename + "_TF_density_hist",
            #         x_hist=self.x_hist,
            #         x_latent_hist=self.x_latent_hist,
            #         x0=x0)