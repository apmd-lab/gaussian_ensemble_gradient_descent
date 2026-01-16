import numpy as np
from scipy.ndimage import gaussian_filter
import gegd.parameter_processing.symmetry_operations as symOp
import gegd.parameter_processing.density_transforms as dtf
import time
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD

class optimizer:
    def __init__(self,
                 Nx,
                 Ny,
                 Nsample,
                 symmetry,
                 periodic,
                 padding,
                 maxiter,
                 high_fidelity_setting,
                 min_feature_size,
                 sigma_RBF,
                 upsample_ratio=1,
                 beta_proj=8,
                 feasible_design_generation_method='brush',
                 brush_shape='circle',
                 cost_obj=None,
                 Nthreads=1,
                 ):

        self.Nx = Nx
        self.Ny = Ny
        self.Nsample = Nsample
        self.symmetry = symmetry
        self.periodic = periodic
        self.padding = padding
        self.maxiter = maxiter
        self.beta_proj = beta_proj
        self.beta_proj_sigma = beta_proj
        self.min_feature_size = min_feature_size
        self.sigma_RBF = sigma_RBF
        self.feasible_design_generation_method = feasible_design_generation_method
        self.upsample_ratio = upsample_ratio
        self.brush_shape = brush_shape
        self.sigma_filter = min_feature_size/2/np.sqrt(2)
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
        
        self.construct_gaussian_covariance()
    
    def construct_gaussian_covariance(self, max_condition_number=1e4):
        self.cov_g = np.zeros((self.Ndim, self.Ndim))
        
        if self.symmetry == 0:
            delta = np.zeros((self.Nx, self.Ny))
            delta[0,0] = 1
            if self.periodic:
                kernel = gaussian_filter(delta, self.sigma_RBF, mode='wrap')
            else:
                kernel = gaussian_filter(delta, self.sigma_RBF, mode='constant')
        
            for i in range(self.Nx):
                for j in range(self.Ny):
                    self.cov_g[self.Ny*i+j,:] = np.roll(kernel, (i, j), axis=(0,1)).reshape(-1)
        
        elif self.symmetry == 1:
            DOF_x = int(np.floor(self.Nx/2 + 0.5))
        
            for i in range(DOF_x):
                for j in range(self.Ny):
                    delta = np.zeros((self.Nx, self.Ny))
                    if self.Nx % 2 == 0:
                        c = 1
                    
                    elif self.Nx % 2 == 1:
                        if i == DOF_x - 1:
                            c = 2
                        else:
                            c = 1
                    
                    delta[i,j] = c
                    delta = symOp.symmetrize(delta[:DOF_x,:], self.symmetry, self.Nx, self.Ny)
                    if self.periodic:
                        kernel = gaussian_filter(delta, self.sigma_RBF, mode='wrap')
                    else:
                        kernel = gaussian_filter(delta, self.sigma_RBF, mode='constant')
                    
                    self.cov_g[i*self.Ny+j,:] = symOp.desymmetrize(kernel, self.symmetry, self.Nx, self.Ny)
        
        elif self.symmetry == 2:
            DOF_x = int(np.floor(self.Nx/2 + 0.5))
            DOF_y = int(np.floor(self.Ny/2 + 0.5))
            
            for i in range(DOF_x):
                for j in range(DOF_y):
                    delta = np.zeros((self.Nx, self.Ny))
                    c = 1
                    if self.Nx % 2 == 1:
                        if i == DOF_x - 1:
                            c *= 2
                    
                    if self.Ny % 2 == 1:
                        if j == DOF_y - 1:
                            c *= 2
                    
                    delta[i,j] = c
                    delta = symOp.symmetrize(delta[:DOF_x,:DOF_y], self.symmetry, self.Nx, self.Ny)
                    if self.periodic:
                        kernel = gaussian_filter(delta, self.sigma_RBF, mode='wrap')
                    else:
                        kernel = gaussian_filter(delta, self.sigma_RBF, mode='constant')
                    
                    self.cov_g[i*DOF_y+j,:] = symOp.desymmetrize(kernel, self.symmetry, self.Nx, self.Ny)
        
        elif self.symmetry == 4:
            triu_ind = np.triu_indices(int(np.floor(self.Nx/2 + 0.5)))
            
            for i in range(self.Ndim):
                delta = np.zeros((self.Nx, self.Ny))
                if self.Nx % 2 == 0:
                    if triu_ind[0][i] == triu_ind[1][i]:
                        c = 2
                    else:
                        c = 1
                
                elif self.Nx % 2 == 1:
                    if i == self.Ndim - 1:
                        c = 8
                    elif triu_ind[1][i] == int((self.Nx - 1)/2):
                        c = 2
                    elif triu_ind[0][i] == triu_ind[1][i]:
                        c = 2
                    else:
                        c = 1
                
                delta[triu_ind[0][i],triu_ind[1][i]] = c
                delta = symOp.symmetrize(delta[triu_ind], self.symmetry, self.Nx, self.Ny)
                if self.periodic:
                    kernel = gaussian_filter(delta, self.sigma_RBF, mode='wrap')
                else:
                    kernel = gaussian_filter(delta, self.sigma_RBF, mode='constant')
                
                self.cov_g[i,:] = symOp.desymmetrize(kernel, self.symmetry, self.Nx, self.Ny)
        
        self.cov_g /= np.max(self.cov_g)
        
        # Determine minimum eps for inversion
        eigvals = np.linalg.eigvalsh(self.cov_g)
        lambda_max = np.max(eigvals)
        lambda_min = np.min(eigvals)
        self.eps = (lambda_max - max_condition_number*lambda_min)/(max_condition_number - 1)
        self.eps = max(self.eps, 0)
    
    def get_cost_samples(self, x):
        # Get Brush Binarized Densities ------------------------------------------------------------
        x_bin = dtf.binarize(x, self.symmetry, self.periodic, self.Nx, self.Ny, self.min_feature_size, self.brush_shape, self.beta_proj, self.sigma_filter,
                             upsample_ratio=self.upsample_ratio, padding=self.padding, method=self.feasible_design_generation_method, Nthreads=self.Nthreads)

        # Sample Modified Cost Function --------------------------------------------------------------
        self.cost_obj.set_accuracy(self.high_fidelity_setting)
        
        f_batch = np.zeros(self.Nsample)
        for n in range(self.Nsample):
            f_batch[n] = self.cost_obj.get_cost(x_bin[n,:], get_grad=False)
        
        return f_batch, x_bin

    def ACMA_ES(
        self,
        c_mu,
        Cov=None,
        B=None,
        D=None,
        eig_eval=None,
        counteval=None,
    ):
        # Set Selection Parameters
        #self.Nsample = 25 #int(4 + np.floor(3 * np.log(self.Ndim)))
        mu = int(np.floor(self.Nsample / 2))
        weights = np.log(mu + 1 / 2) - np.log(np.arange(mu) + 1)
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)

        # Initialize Dynamic Internal Parameters
        if Cov is None:
            Cov = self.cov_g.copy()
            Cov = np.triu(Cov) + np.triu(Cov, k=1).T
        if D is None and B is None:
            [D, B] = np.linalg.eigh(Cov + self.eps*np.identity(self.Ndim))
            D = np.sqrt(np.diag(D))
        eig_eval = eig_eval if eig_eval is not None else 0

        counteval = counteval if counteval is not None else 0

        while True:
            t1 = time.time()

            # Sample Candidates
            arz = np.random.normal(size=(self.Ndim, self.Nsample))
            arx = B @ D @ arz
            
            # Cost Evaluation
            cost, x_bin = self.get_cost_samples(arx.T)
            sorted_index = np.argsort(cost)
            cost = cost[sorted_index]
            arx = arx[:,sorted_index]
            x_bin = x_bin[sorted_index,:]
            counteval += self.Nsample

            if self.x_latent_hist is None:
                self.x_latent_hist = arx[:,0].copy()
            else:
                self.x_latent_hist = np.vstack((self.x_latent_hist, arx[:,0]))
            
            if self.best_cost_hist.size == 0:
                new_best = True
                self.best_cost_hist = np.append(self.best_cost_hist, cost[0])
            else:
                new_best = self.best_cost_hist[-1] > cost[0]
                self.best_cost_hist = np.append(self.best_cost_hist, np.min((self.best_cost_hist[-1], cost[0])))
                
            if self.best_x_hist is None:
                self.best_x_hist = x_bin[0,:].copy()
            else:
                if new_best:
                    self.best_x_hist = np.vstack((self.best_x_hist, x_bin[0,:]))
                else:
                    if self.best_x_hist.ndim == 1:
                        self.best_x_hist = np.vstack((self.best_x_hist, self.best_x_hist))
                    else:
                        self.best_x_hist = np.vstack((self.best_x_hist, self.best_x_hist[-1,:]))
            
            self.cost_ensemble_hist = np.append(self.cost_ensemble_hist, np.mean(cost))
            self.cost_ensemble_sigma_hist = np.append(self.cost_ensemble_sigma_hist, np.std(cost))
            
            if comm.rank == 0:
                print('    | %4d |  %4d |  %9.2f |  %8.3f |' %(
                    self.n_iter,
                    self.Nsample,
                    np.mean(cost),
                    self.best_cost_hist[-1],
                ), end='', flush=True)

            self.save_data(
                B=B,
                D=D,
                Cov=Cov,
                eig_eval=eig_eval,
                counteval=counteval,
            )

            if self.n_iter > self.maxiter:
                self.n_iter += 1
                break

            # Update Covariance Matrix
            Cov = (1 - c_mu) * Cov \
                + c_mu * (B @ D @ arz[:,:mu]) @ np.diag(weights) @ (B @ D @ arz[:,:mu]).T

            # Update B and D
            if counteval - eig_eval > self.Nsample / (c_mu * self.Ndim * 10):
                eig_eval = counteval
                Cov = np.triu(Cov) + np.triu(Cov, k=1).T
                [D, B] = np.linalg.eigh(Cov + self.eps*np.identity(self.Ndim))
                D = np.sqrt(np.diag(D))
            
            t2 = time.time()
            t_rem = (t2 - t1)*(self.maxiter - self.n_iter + 1)/3600

            if comm.rank == 0:
                print('   %5.2f   |' % t_rem, flush=True)
            
            self.n_iter += 1

    def run(self, n_seed, output_filename, eta=None, load_data=False):
        if comm.rank == 0:
            print('### ACMA-ES (seed = ' + str(n_seed) + ')\n', flush=True)
    
        np.random.seed(n_seed)
        self.output_filename = output_filename
        
        if load_data:
            data_file1 = output_filename + "_ACMA_ES_results.npz"
            data_file2 = output_filename + "_ACMA_ES_density_hist.npz"
                
            with np.load(data_file1) as data:
                self.n_iter = data['n_iter']
                
                self.cost_ensemble_hist = data['cost_ensemble_hist'][:self.n_iter]
                self.best_cost_hist = data['best_cost_hist'][:self.n_iter]
                self.cost_ensemble_sigma_hist = data['cost_ensemble_sigma_hist'][:self.n_iter]

                eig_eval = data['eig_eval']
                counteval = data['counteval']
                
            with np.load(data_file2) as data:
                self.x_latent_hist = data['x_latent_hist'][:self.n_iter,:]
                self.best_x_hist = data['best_x_hist'][:self.n_iter,:]
                
                B = data['B']
                D = data['D']
                Cov = data['Cov']

        else:
            self.n_iter = 0
            
            self.x_latent_hist = None
            self.best_x_hist = None
            self.cost_ensemble_hist = np.zeros(0)
            self.best_cost_hist = np.zeros(0)
            self.cost_ensemble_sigma_hist = np.zeros(0)
            
            Cov = None
            B = None
            D = None
            eig_eval = None
            counteval = None
        
        if comm.rank == 0:
            print('    | Iter |   N   |  Cost Ens  | Cost Best | t_rem(hr) |',
                  flush=True)

        self.ACMA_ES(
            c_mu=eta,
            Cov=Cov,
            B=B,
            D=D,
            eig_eval=eig_eval,
            counteval=counteval,
        )
        
        self.save_data()
    
    def save_data(
        self,
        B=None,
        D=None,
        Cov=None,
        eig_eval=None,
        counteval=None,
    ):

        if comm.rank == 0:
            if B is None:
                with np.load(self.output_filename + "_ACMA_ES_density_hist.npz") as data:
                    B = data['B']
                    D = data['D']
                    Cov = data['Cov']

            if eig_eval is None:
                with np.load(self.output_filename + "_ACMA_ES_results.npz") as data:
                    eig_eval = data['eig_eval']
                    counteval = data['counteval']
        
            # Customize below
            np.savez(self.output_filename + "_ACMA_ES_results",
                cost_ensemble_hist=self.cost_ensemble_hist,
                cost_ensemble_sigma_hist=self.cost_ensemble_sigma_hist,
                best_cost_hist=self.best_cost_hist,
                n_iter=self.n_iter,
                eig_eval=eig_eval,
                counteval=counteval,
            )
                        
            np.savez(self.output_filename + "_ACMA_ES_density_hist",
                x_latent_hist=self.x_latent_hist,
                best_x_hist=self.best_x_hist,
                B=B,
                D=D,
                Cov=Cov,
            )