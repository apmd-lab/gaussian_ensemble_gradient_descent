import numpy as np
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
                 symmetry,
                 periodic,
                 padding,
                 maxiter,
                 high_fidelity_setting,
                 brush_size,
                 sigma_ensemble_max=1.0,
                 upsample_ratio=1,
                 beta_proj=8,
                 brush_shape='circle',
                 covariance_type='constant',
                 coeff_exp=5,
                 cost_threshold=0,
                 cost_obj=None,
                 Nthreads=1,
                 ):
        
        self.Nx = Nx
        self.Ny = Ny
        self.symmetry = symmetry
        self.periodic = periodic
        self.padding = padding
        self.maxiter = maxiter
        self.beta_proj = beta_proj
        self.beta_proj_sigma = beta_proj
        self.brush_size = brush_size
        self.sigma_ensemble_max = sigma_ensemble_max
        self.upsample_ratio = upsample_ratio
        self.brush_shape = brush_shape
        self.sigma_filter = brush_size/2/np.sqrt(2)
        self.sigma_RBF = brush_size/2/np.sqrt(2)
        self.covariance_type = covariance_type
        self.high_fidelity_setting = high_fidelity_setting
        self.coeff_exp = coeff_exp
        self.cost_threshold = cost_threshold
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
        
        if self.covariance_type == 'constant':
            self.Nsigma = 1
            
        elif self.covariance_type == 'diagonal':
            self.Nsigma = self.Ndim
    
    def construct_cov_matrix(self, sigma):
        if self.covariance_type == 'constant':
            cov = sigma**2*np.identity(self.Ndim)
            cov_inv = (1/sigma**2)*np.identity(self.Ndim)
        
        elif self.covariance_type == 'diagonal':
            cov = np.diag(sigma**2)
            cov_inv = np.diag(1/sigma**2)
        
        return cov, cov_inv
    
    def mu_derivative(self, dx, cov_inv, p):
        
        return (cov_inv @ dx)*p
    
    def sigma_derivative(self, dx, sigma, cov_inv, p):
        if self.covariance_type == 'constant':
            grad = ((dx.reshape(1,-1) @ dx.reshape(-1,1))/sigma**3 - self.Ndim/sigma)*p
        
        elif self.covariance_type == 'diagonal':
            grad = np.diag(cov_inv @ dx.reshape(-1,1) @ dx.reshape(1,-1) @ np.sqrt(cov_inv) - np.sqrt(cov_inv))*p
        
        return grad
    
    def ensemble_jacobian(self, x0, dx, sigma_ensemble, test_function):
        x0 = x0[:self.Ndim]
        cov, cov_inv = self.construct_cov_matrix(sigma_ensemble)
        
        # Get Brush Binarized Densities ------------------------------------------------------------
        x_sample = dtf.binarize(x0, self.symmetry, self.periodic, self.Nx, self.Ny, self.brush_size, self.brush_shape, self.beta_proj, self.sigma_filter, dx=dx,
                                upsample_ratio=self.upsample_ratio, padding=self.padding, Nthreads=self.Nthreads)
        
        # Sample Modified Cost Function --------------------------------------------------------------
        self.cost_obj.set_accuracy(self.high_fidelity_setting)
        
        f = np.zeros(self.Nensemble)
        f_logDeriv = np.zeros((self.Nensemble, self.Ndim+self.Nsigma))
        for n in range(self.Nensemble):
            f_temp = self.cost_obj.get_cost(x_sample[n,:], False)
            f_shifted = (f_temp - self.cost_threshold)/(self.cost_threshold + 1)
            f[n] = -np.exp(-self.coeff_exp*f_shifted)
            f_logDeriv[n,:self.Ndim] = self.mu_derivative(dx[n,:], cov_inv, f[n])
            f_logDeriv[n,self.Ndim:] = self.sigma_derivative(dx[n,:], sigma_ensemble, cov_inv, f[n])

        if test_function:
            x_fp0, x_b0 = dtf.binarize(x0.reshape(1,-1), self.symmetry, self.periodic, self.Nx, self.Ny, self.brush_size, self.brush_shape, self.beta_proj, self.sigma_filter,
                                       upsample_ratio=self.upsample_ratio, output_details=True, padding=self.padding, Nthreads=self.Nthreads)
            f0, jac_b_STE_sym = self.cost_obj.get_cost(x_b0, True)
            jac_b_STE_sym = jac_b_STE_sym.reshape(self.Nx, self.Ny)
        else:
            x_fp0 = np.zeros_like(x0)
            x_b0 = np.zeros_like(x0)
            f0 = 0
        
        # Get Best Cost -----------------------------------------------------------------------------------
        f_best = (-np.log(-np.min(f))/self.coeff_exp)*(self.cost_threshold + 1) + self.cost_threshold
        x_best = x_sample[np.argmin(f),:].copy()
        
        # Control Variate Estimation of the gradient --------------------------------------------------------------
        jac_fp_ensemble = f_logDeriv.copy()
        jac_fp_ensemble = np.mean(jac_fp_ensemble, axis=0)

        jac_mu_fp_ensemble_sym = symOp.symmetrize_jacobian(jac_fp_ensemble[:self.Ndim], self.symmetry, self.Nx, self.Ny)
        if self.Nsigma > 1:
            jac_sigma_fp_ensemble_sym = symOp.symmetrize_jacobian(jac_fp_ensemble[self.Ndim:], self.symmetry, self.Nx, self.Ny)
        
        if test_function:
            jac_latent_STE = dtf.backprop_filter_and_project(jac_b_STE_sym, x0, self.symmetry, self.periodic, self.Nx, self.Ny, self.brush_size, self.sigma_filter, self.beta_proj, padding=self.padding)
        
        jac_latent_ensemble = jac_fp_ensemble.copy()
        jac_latent_ensemble[:self.Ndim] = dtf.backprop_filter_and_project(jac_mu_fp_ensemble_sym, x0, self.symmetry, self.periodic, self.Nx, self.Ny, self.brush_size, self.sigma_filter, self.beta_proj, padding=self.padding)
        if self.Nsigma > 1:
            jac_latent_ensemble[self.Ndim:] = dtf.backprop_filter_and_project(jac_sigma_fp_ensemble_sym, x0, self.symmetry, self.periodic, self.Nx, self.Ny, self.brush_size, self.sigma_filter, self.beta_proj_sigma, padding=self.padding)
        
        if test_function:
            cost_all = (f0, np.mean(f_est))
            density_all = (x_fp0, x_b0)
            grad_STE = (jac_b_STE_sym, jac_latent_STE)
            grad_ensemble = (jac_mu_fp_ensemble_sym, jac_latent_ensemble)
        
            return cost_all, density_all, grad_STE, grad_ensemble
        else:
            return f0, x_b0, np.mean(f), np.std(f), jac_latent_ensemble, f_best, x_best

    def NES(
        self,
        x_bounded,
        lb,
        ub,
        eta_NES,
    ):
    
        # Dummy Variables
        x = -np.log((ub - lb)/(x_bounded - lb) - 1)
        x[ub==lb] = ub[ub==lb]
        
        while True:
            t1 = time.time()
            
            x_bounded = lb + (ub - lb)/(1 + np.exp(-x))
            
            sigma_ensemble = x_bounded[self.Ndim:]
            if self.Nsigma == 1:
                sigma_fp = sigma_ensemble.copy()
            else:
                sigma_scaled = 2*(sigma_ensemble - lb[self.Ndim:])/(ub[self.Ndim:] - lb[self.Ndim:]) - 1
                sigma_fp_scaled = dtf.filter_and_project(sigma_scaled, self.symmetry, self.periodic, self.Nx, self.Ny, self.brush_size, self.sigma_filter, self.beta_proj_sigma, padding=self.padding)
                sigma_fp = (sigma_fp_scaled + 1)*(ub[self.Ndim:] - lb[self.Ndim:])/2 + lb[self.Ndim:]
            cov, cov_inv = self.construct_cov_matrix(sigma_fp)
            
            dx = np.random.multivariate_normal(np.zeros(self.Ndim), cov, size=self.Nensemble)
            
            loss, x_bin0, loss_mean, loss_std, jac, f_best, x_best = self.ensemble_jacobian(
                x_bounded,
                dx,
                sigma_fp,
                False,
            )
            
            # Precondition Gradients with the Inverse Fisher Information Matrix
            jac[:self.Ndim] = cov @ jac[:self.Ndim]
            jac[self.Ndim:] = sigma_fp**2*jac[self.Ndim:]/2
            
            if self.Nsigma > 1:
                jac[self.Ndim:] *= 2/(ub[self.Ndim:] - lb[self.Ndim:]) # accounts for covariance rescaling
            
            jac_save = jac.copy()

            jac[ub==lb] = 0
            jac *= np.exp(-x)*(ub - lb)/(1 + np.exp(-x))**2
            
            if self.x_latent_hist is None:
                self.x_latent_hist = x_bounded[:self.Ndim].copy()
            else:
                self.x_latent_hist = np.vstack((self.x_latent_hist, x_bounded[:self.Ndim]))
                
            if self.x_hist is None:
                self.x_hist = x_bin0.copy()
            else:
                self.x_hist = np.vstack((self.x_hist, x_bin0))
            
            if self.best_cost_hist.size == 0:
                new_best = True
                self.best_cost_hist = np.append(self.best_cost_hist, f_best)
            else:
                new_best = self.best_cost_hist[-1] > f_best
                self.best_cost_hist = np.append(self.best_cost_hist, np.min((self.best_cost_hist[-1], f_best)))
                
            if self.best_x_hist is None:
                self.best_x_hist = x_best.copy()
            else:
                if new_best:
                    self.best_x_hist = np.vstack((self.best_x_hist, x_best))
                else:
                    if self.best_x_hist.ndim == 1:
                        self.best_x_hist = np.vstack((self.best_x_hist, self.best_x_hist))
                    else:
                        self.best_x_hist = np.vstack((self.best_x_hist, self.best_x_hist[-1,:]))
            
            self.cost_ensemble_hist = np.append(self.cost_ensemble_hist, loss_mean)
            self.cost_ensemble_sigma_hist = np.append(self.cost_ensemble_sigma_hist, loss_std)
            
            if self.Nsigma == 1:
                self.sigma_ensemble_hist = np.append(self.sigma_ensemble_hist, sigma_ensemble)
            else:
                if self.sigma_ensemble_hist is None:
                    self.sigma_ensemble_hist = sigma_ensemble.copy()
                else:
                    self.sigma_ensemble_hist = np.vstack((self.sigma_ensemble_hist, sigma_ensemble))
                
                if self.sigma_fp_hist is None:
                    self.sigma_fp_hist = sigma_fp.copy()
                else:
                    self.sigma_fp_hist = np.vstack((self.sigma_fp_hist, sigma_fp))
            
            t2 = time.time()
            t_rem = (t2 - t1)*(self.maxiter - self.n_iter + 1)/3600
            
            if comm.rank == 0:
                if self.Nsigma == 1:
                    print('    | %4d |  %4d | %7.5f |  %9.2f |  %8.3f |   %5.2f   |' %(
                        self.n_iter,
                        self.Nensemble,
                        sigma_fp[0],
                        -np.log(-loss_mean)/self.coeff_exp,
                        self.best_cost_hist[-1],
                        t_rem,
                    ), end='', flush=True)
                
                else:
                    print('    | %4d |  %4d | %7.5f | %7.5f |  %9.2f |  %8.3f |   %5.2f   |' %(
                        self.n_iter,
                        self.Nensemble,
                        np.min(sigma_fp),
                        np.max(sigma_fp),
                        loss_mean,
                        self.best_cost_hist[-1],
                        t_rem,
                    ), end='', flush=True)

            self.save_data(x_bounded=x_bounded)

            if self.n_iter > self.maxiter:
                self.n_iter += 1
                break

            # Update Variables
            x -= eta_NES*jac
            
            self.n_iter += 1

    def run(self, n_seed, output_filename, x0=None, eta_mu=0.01, eta_sigma=1.0, load_data=False):
        if comm.rank == 0:
            print('### Natural Evolution Strategy (seed = ' + str(n_seed) + ')\n', flush=True)
    
        np.random.seed(n_seed)
        self.output_filename = output_filename
        
        sigma_ensemble = self.sigma_ensemble_max/2

        lb = np.zeros(self.Ndim+self.Nsigma)
        lb[:self.Ndim] = -1
        lb[self.Ndim:] = 0.99*self.sigma_ensemble_max/2
        ub = np.zeros(self.Ndim+self.Nsigma)
        ub[:self.Ndim] = 1
        ub[self.Ndim:] = 1.01*self.sigma_ensemble_max/2
        #ub[self.Ndim:] = 10.0
        
        if load_data:
            data_file1 = output_filename + "_NES_results.npz"
            data_file2 = output_filename + "_NES_density_hist.npz"
                
            with np.load(data_file1) as data:
                self.n_iter = data['n_iter']
                
                self.cost_ensemble_hist = data['cost_ensemble_hist'][:self.n_iter]
                self.best_cost_hist = data['best_cost_hist'][:self.n_iter]
                self.cost_ensemble_sigma_hist = data['cost_ensemble_sigma_hist'][:self.n_iter]
                if self.Nsigma == 1:
                    self.sigma_ensemble_hist = data['sigma_ensemble_hist'][:self.n_iter]
                
            with np.load(data_file2) as data:
                self.x_latent_hist = data['x_latent_hist'][:self.n_iter,:]
                self.x_hist = data['x_hist'][:self.n_iter,:]
                self.best_x_hist = data['best_x_hist'][:self.n_iter,:]
                if self.Nsigma > 1:
                    self.sigma_ensemble_hist = data['sigma_ensemble_hist'][:self.n_iter,:]
                    self.sigma_fp_hist = data['sigma_fp_hist'][:self.n_iter,:]
                x0 = data['x_bounded']
            
        else:
            self.n_iter = 0
            
            self.x_latent_hist = None
            self.x_hist = None
            self.best_x_hist = None
            self.cost_ensemble_hist = np.zeros(0)
            self.best_cost_hist = np.zeros(0)
            self.cost_ensemble_sigma_hist = np.zeros(0)
            if self.Nsigma == 1:
                self.sigma_ensemble_hist = np.zeros(0)
            else:
                self.sigma_ensemble_hist = None
                self.sigma_fp_hist = None
            
            if x0 is None:
                # Initial Structure
                x0 = np.hstack((np.zeros(self.Ndim), sigma_ensemble*np.ones(self.Nsigma)))
        
        if comm.rank == 0:
            if self.Nsigma == 1:
                print('    | Iter |   N   |  sigma  |  Cost Ens  | Cost Best | t_rem(hr) |',
                      flush=True)
            else:
                print('    | Iter |   N   | sig_min | sig_max |  Cost Ens  | Cost Best | t_rem(hr) |',
                      flush=True)

        eta_NES = np.ones(self.Ndim + self.Nsigma)
        eta_NES[:self.Ndim] = eta_mu
        eta_NES[self.Ndim:] = eta_sigma
        self.NES(x0, lb, ub, eta_NES)
        
        self.save_data()
    
    def save_data(self, x_bounded=None):
        if comm.rank == 0:
            if x_bounded is None:
                with np.load(self.output_filename + "_NES_density_hist.npz") as data:
                    x_bounded = data['x_bounded']
        
            # Customize below
            if self.Nsigma == 1:
                np.savez(self.output_filename + "_NES_results",
                         cost_ensemble_hist=self.cost_ensemble_hist,
                         cost_ensemble_sigma_hist=self.cost_ensemble_sigma_hist,
                         best_cost_hist=self.best_cost_hist,
                         sigma_ensemble_hist=self.sigma_ensemble_hist,
                         n_iter=self.n_iter,
                         )
                         
                np.savez(self.output_filename + "_NES_density_hist",
                         x_hist=self.x_hist,
                         x_latent_hist=self.x_latent_hist,
                         best_x_hist=self.best_x_hist,
                         x_bounded=x_bounded,
                         )
            
            else:
                np.savez(self.output_filename + "_NES_results",
                         cost_ensemble_hist=self.cost_ensemble_hist,
                         cost_ensemble_sigma_hist=self.cost_ensemble_sigma_hist,
                         best_cost_hist=self.best_cost_hist,
                         n_iter=self.n_iter,
                         )
                         
                np.savez(self.output_filename + "_NES_density_hist",
                         x_hist=self.x_hist,
                         sigma_ensemble_hist=self.sigma_ensemble_hist,
                         sigma_fp_hist=self.sigma_fp_hist,
                         x_latent_hist=self.x_latent_hist,
                         best_x_hist=self.best_x_hist,
                         x_bounded=x_bounded,
                         )