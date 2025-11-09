import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interpn
from itertools import product
import gegd.parameter_processing.symmetry_operations as symOp
import gegd.parameter_processing.density_transforms as dtf
import time

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
                 t_low_fidelity,
                 t_high_fidelity,
                 t_iteration,
                 high_fidelity_setting,
                 low_fidelity_setting,
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
        self.t_low_fidelity = t_low_fidelity
        self.t_high_fidelity = t_high_fidelity
        self.t_iteration = t_iteration
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
        self.low_fidelity_setting = low_fidelity_setting
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
            
        elif self.covariance_type == 'gaussian_constant':
            self.Nsigma = 1
            self.construct_gaussian_covariance()
            
        elif self.covariance_type == 'gaussian_diagonal':
            self.Nsigma = self.Ndim
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
        eps = (lambda_max - max_condition_number*lambda_min)/(max_condition_number - 1)
        eps = max(eps, 0)
        
        self.cov_g = self.cov_g + eps*np.identity(self.Ndim)
        self.cov_g_inv = np.linalg.inv(self.cov_g)
    
    def construct_cov_matrix(self, sigma):
        if self.covariance_type == 'constant':
            cov = sigma**2*np.identity(self.Ndim)
            cov_inv = (1/sigma**2)*np.identity(self.Ndim)
        
        elif self.covariance_type == 'diagonal':
            cov = np.diag(sigma**2)
            cov_inv = np.diag(1/sigma**2)
        
        elif self.covariance_type == 'gaussian_constant':
            cov = sigma**2*self.cov_g
            cov_inv = (1/sigma**2)*self.cov_g_inv
        
        elif self.covariance_type == 'gaussian_diagonal':
            cov = np.diag(sigma) @ self.cov_g @ np.diag(sigma)
            cov_inv = np.diag(1/sigma) @ self.cov_g_inv @ np.diag(1/sigma)
        
        return cov, cov_inv
    
    def mu_derivative(self, dx, cov_inv, p):
        
        return (cov_inv @ dx)*p
    
    def sigma_derivative(self, dx, sigma, cov_inv, p):
        if self.covariance_type == 'constant':
            grad = ((dx.reshape(1,-1) @ dx.reshape(-1,1))/sigma**3 - self.Ndim/sigma)*p
        
        elif self.covariance_type == 'diagonal':
            grad = np.diag(cov_inv @ dx.reshape(-1,1) @ dx.reshape(1,-1) @ np.sqrt(cov_inv) - np.sqrt(cov_inv))*p
            
        elif self.covariance_type == 'gaussian_constant':
            grad = (1/sigma)*(dx.reshape(1,-1) @ cov_inv @ dx.reshape(-1,1) - self.Ndim)*p
        
        elif self.covariance_type == 'gaussian_diagonal':
            z = cov_inv @ dx
            Bz = (self.cov_g*sigma[np.newaxis,:]) @ z
            A_diag = z*Bz
        
            grad = (A_diag - 1/sigma)*p
        
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
        
        # Sample Control Variate Function ----------------------------------------------------------------
        self.cost_obj.set_accuracy(self.low_fidelity_setting)
        
        f_ctrl_all = np.zeros(self.r_CV*self.Nensemble)
        f_ctrl_logDeriv_all = np.zeros((self.r_CV*self.Nensemble, self.Ndim+self.Nsigma))
        for n in range(self.r_CV*self.Nensemble):
            f_temp = self.cost_obj.get_cost(x_sample[n,:], False)
            f_shifted = (f_temp - self.cost_threshold)/(self.cost_threshold + 1)
            f_ctrl_all[n] = -np.exp(-self.coeff_exp*f_shifted)
            f_ctrl_logDeriv_all[n,:self.Ndim] = self.mu_derivative(dx[n,:], cov_inv, f_ctrl_all[n])
            f_ctrl_logDeriv_all[n,self.Ndim:] = self.sigma_derivative(dx[n,:], sigma_ensemble, cov_inv, f_ctrl_all[n])
        
        # Expectation of the Control Variate Function -----------------------------------------------------
        ctrlVarExp_f = np.mean(f_ctrl_all)
        ctrlVarExp_mu = np.mean(f_ctrl_logDeriv_all[:,:self.Ndim], axis=0)
        ctrlVarExp_sigma = np.mean(f_ctrl_logDeriv_all[:,self.Ndim:], axis=0)

        # Control Variate Coefficient (expectation) --------------------------------------------------------------------
        f_ctrl = f_ctrl_all[:self.Nensemble]
        mean_f = np.mean(f)
        mean_f_ctrl = np.mean(f_ctrl)
        
        cov_f_f_ctrl = (1/(self.Nensemble - 1))*np.sum((f - mean_f)*(f_ctrl - mean_f_ctrl))
        var_f = (1/(self.Nensemble - 1))*np.sum((f - mean_f)**2)
        var_ctrl = (1/(self.Nensemble - 1))*np.sum((f_ctrl - mean_f_ctrl)**2)

        ctrlVarCoeff_f = cov_f_f_ctrl/var_ctrl
        corr_f = cov_f_f_ctrl/np.sqrt(var_f*var_ctrl)
        
        # Control Variate Coefficient (expectation gradient wrt mu) --------------------------------------------------------------------
        f_mu = f_logDeriv[:,:self.Ndim]
        f_ctrl_mu = f_ctrl_logDeriv_all[:self.Nensemble,:self.Ndim]
        mean_f_mu = np.mean(f_mu, axis=0)
        mean_f_ctrl_mu = np.mean(f_ctrl_mu, axis=0)
        
        cov_f_f_ctrl = (1/(self.Nensemble - 1))*np.sum((f_mu - mean_f_mu)*(f_ctrl_mu - mean_f_ctrl_mu), axis=0)
        var_f = (1/(self.Nensemble - 1))*np.sum((f_mu - mean_f_mu)**2, axis=0)
        var_ctrl = (1/(self.Nensemble - 1))*np.sum((f_ctrl_mu - mean_f_ctrl_mu)**2, axis=0)

        weights = np.ones(self.Ndim) #1/np.hstack((sigma_ensemble, sigma_ensemble))**2
        ctrlVarCoeff_mu = np.average(cov_f_f_ctrl, weights=weights)/np.average(var_ctrl, weights=weights)
        corr_mu = cov_f_f_ctrl/np.sqrt(var_f*var_ctrl)
        
        # Control Variate Coefficient (expectation gradient wrt sigma) --------------------------------------------------------------------
        f_sigma = f_logDeriv[:,self.Ndim:]
        f_ctrl_sigma = f_ctrl_logDeriv_all[:self.Nensemble,self.Ndim:]
        mean_f_sigma = np.mean(f_sigma, axis=0)
        mean_f_ctrl_sigma = np.mean(f_ctrl_sigma, axis=0)
        
        cov_f_f_ctrl = (1/(self.Nensemble - 1))*np.sum((f_sigma - mean_f_sigma)*(f_ctrl_sigma - mean_f_ctrl_sigma), axis=0)
        var_f = (1/(self.Nensemble - 1))*np.sum((f_sigma - mean_f_sigma)**2, axis=0)
        var_ctrl = (1/(self.Nensemble - 1))*np.sum((f_ctrl_sigma - mean_f_ctrl_sigma)**2, axis=0)

        weights = np.ones(self.Nsigma) #1/np.hstack((sigma_ensemble, sigma_ensemble))**2
        ctrlVarCoeff_sigma = np.average(cov_f_f_ctrl, weights=weights)/np.average(var_ctrl, weights=weights)
        corr_sigma = cov_f_f_ctrl/np.sqrt(var_f*var_ctrl)
        
        # Control Variate Estimation of the expectation --------------------------------------------------------------
        if self.r_CV == 1:
            f_est = f.copy()
        else:
            f_est = f - ctrlVarCoeff_f*(f_ctrl - ctrlVarExp_f)
        
        # Control Variate Estimation of the gradient --------------------------------------------------------------
        jac_fp_ensemble = f_logDeriv.copy()
        if self.r_CV > 1:
            jac_fp_ensemble[:,:self.Ndim] -= ctrlVarCoeff_mu*(f_ctrl_mu - ctrlVarExp_mu)
            jac_fp_ensemble[:,self.Ndim:] -= ctrlVarCoeff_sigma*(f_ctrl_sigma - ctrlVarExp_sigma)
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
            return f0, x_b0, np.mean(f), np.std(f), np.mean(f_ctrl_all), np.std(f_ctrl_all), jac_latent_ensemble, ctrlVarCoeff_mu, ctrlVarCoeff_sigma, corr_f, corr_mu, corr_sigma, f_best, x_best
    
    def ADAM(self,
             x_bounded,
             lb,
             ub,
             beta_ADAM1,
             beta_ADAM2,
             eta_ADAM,
             jac_mean=None,
             jac_var=None,
             adam_iter=None,
             x_bounded_norm_ref=None,
             ):
    
        # Dummy Variables
        x = -np.log((ub - lb)/(x_bounded - lb) - 1)
        x[ub==lb] = ub[ub==lb]
        
        if jac_mean is None:
            jac_mean = np.zeros_like(x)
        if jac_var is None:
            jac_var = np.zeros_like(x)
        if adam_iter is None:
            adam_iter = 0
            
        while True:
            t1 = time.time()
            adam_iter += 1
            
            # Determine optimal ensemble sizes for the high and low-fidelity simulations given the target iteration time
            N = np.arange(int(np.floor(self.t_iteration/self.t_high_fidelity))) + 1
            N = N[N>=5]
            r_CV = np.floor(self.t_iteration/(self.t_low_fidelity*N) - self.t_high_fidelity/self.t_low_fidelity).astype(np.int32)
                        
            N = N[(r_CV>0)*(r_CV<=100)]
            r_CV = r_CV[(r_CV>0)*(r_CV<=100)]
            
            if self.corr_mu_hist is None:
                corr_mean = 0
            elif self.corr_mu_hist.ndim == 1:
                corr_mean = np.mean(self.corr_mu_hist)
            else:
                corr_mean = np.mean(self.corr_mu_hist[-1,:])
            
            var_reduction = np.zeros(N.size)
            var_reduction[r_CV>0] = (1 - ((r_CV[r_CV>0] - 1)/r_CV[r_CV>0])*corr_mean**2)/N[r_CV>0]
            var_reduction[r_CV<=0] = 1/N[r_CV<=0]
            ind_opt = np.nanargmin(var_reduction)
            self.Nensemble = N[ind_opt]
            self.r_CV = r_CV[ind_opt]
            
            var_reduction = (1 - ((self.r_CV - 1)/self.r_CV)*corr_mean**2)/self.Nensemble
            
            x_bounded = lb + (ub - lb)/(1 + np.exp(-x))
            
            sigma_ensemble = x_bounded[self.Ndim:]
            if self.Nsigma == 1:
                sigma_fp = sigma_ensemble.copy()
            else:
                sigma_scaled = 2*(sigma_ensemble - lb[self.Ndim:])/(ub[self.Ndim:] - lb[self.Ndim:]) - 1
                sigma_fp_scaled = dtf.filter_and_project(sigma_scaled, self.symmetry, self.periodic, self.Nx, self.Ny, self.brush_size, self.sigma_filter, self.beta_proj_sigma, padding=self.padding)
                sigma_fp = (sigma_fp_scaled + 1)*(ub[self.Ndim:] - lb[self.Ndim:])/2 + lb[self.Ndim:]
            cov, cov_inv = self.construct_cov_matrix(sigma_fp)
            
            dx = np.random.multivariate_normal(np.zeros(self.Ndim), cov, size=self.r_CV*self.Nensemble)
            
            loss, x_bin0, loss_mean, loss_std, loss_ctrl_mean, loss_ctrl_std, jac, ctrlVarCoeff_mu, ctrlVarCoeff_sigma, corr_f, corr_mu, corr_sigma, f_best, x_best = self.ensemble_jacobian(x_bounded,
                                                                                                                                                                                             dx,
                                                                                                                                                                                             sigma_fp,
                                                                                                                                                                                             False)
            
            if self.Nsigma > 1:
                jac[self.Ndim:] *= 2/(ub[self.Ndim:] - lb[self.Ndim:]) # accounts for covariance rescaling
            
            jac_save = jac.copy()

            jac[ub==lb] = 0
            jac *= np.exp(-x)*(ub - lb)/(1 + np.exp(-x))**2
            
            self.N_high_fidelity_hist = np.append(self.N_high_fidelity_hist, self.Nensemble)
            self.N_low_fidelity_hist = np.append(self.N_low_fidelity_hist, self.r_CV*self.Nensemble)
            self.var_reduction_hist = np.append(self.var_reduction_hist, var_reduction)
            self.N_eff_hist = np.append(self.N_eff_hist, int(1/var_reduction))
            
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
            self.cost_ensemble_ctrl_hist = np.append(self.cost_ensemble_ctrl_hist, loss_ctrl_mean)
            self.cost_ensemble_ctrl_sigma_hist = np.append(self.cost_ensemble_ctrl_sigma_hist, loss_ctrl_std)
            
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
            
            self.ctrlVarCoeff_mu_hist = np.append(self.ctrlVarCoeff_mu_hist, ctrlVarCoeff_mu)
            self.ctrlVarCoeff_sigma_hist = np.append(self.ctrlVarCoeff_sigma_hist, ctrlVarCoeff_sigma)
            self.corr_f_hist = np.append(self.corr_f_hist, corr_f)
            
            if self.corr_mu_hist is None:
                self.corr_mu_hist = corr_mu.copy()
            else:
                self.corr_mu_hist = np.vstack((self.corr_mu_hist, corr_mu))
            if self.corr_sigma_hist is None:
                self.corr_sigma_hist = corr_sigma.copy()
            else:
                self.corr_sigma_hist = np.vstack((self.corr_sigma_hist, corr_sigma))
        
            t2 = time.time()
            t_rem = (t2 - t1)*(self.maxiter - self.n_iter + 1)/3600
            
            if comm.rank == 0:
                if self.Nsigma == 1:
                    print('    | %4d |  %4d |   %4d  | %7.5f |  %4d | %7.5f |  %5.2f  | %7.4f | %7.4f |  %9.2f |  %9.2f  |  %8.3f |' %(self.n_iter,
                                                                                                                                       self.Nensemble,
                                                                                                                                       self.r_CV*self.Nensemble,
                                                                                                                                       var_reduction,
                                                                                                                                       int(1/var_reduction),
                                                                                                                                       sigma_fp,
                                                                                                                                       ctrlVarCoeff_mu,
                                                                                                                                       np.mean(corr_f),
                                                                                                                                       np.mean(corr_mu),
                                                                                                                                       -np.log(-loss_mean)/self.coeff_exp,
                                                                                                                                       -np.log(-loss_ctrl_mean)/self.coeff_exp,
                                                                                                                                       self.best_cost_hist[-1]),
                                                                                                                                       end='', flush=True)
                
                else:
                    print('    | %4d |  %4d |   %4d  | %7.5f |  %4d | %7.5f | %7.5f |  %5.2f  | %7.4f | %7.4f |  %9.2f | %7.2f |  %9.2f  |   %7.2f  |  %8.3f |' %(self.n_iter,
                                                                                                                                                                  self.Nensemble,
                                                                                                                                                                  self.r_CV*self.Nensemble,
                                                                                                                                                                  var_reduction,
                                                                                                                                                                  int(1/var_reduction),
                                                                                                                                                                  np.min(sigma_fp),
                                                                                                                                                                  np.max(sigma_fp),
                                                                                                                                                                  ctrlVarCoeff_mu,
                                                                                                                                                                  np.mean(corr_f),
                                                                                                                                                                  np.mean(corr_mu),
                                                                                                                                                                  loss_mean,
                                                                                                                                                                  loss_std,
                                                                                                                                                                  loss_ctrl_mean,
                                                                                                                                                                  loss_ctrl_std,
                                                                                                                                                                  self.best_cost_hist[-1]),
                                                                                                                                                                  end='', flush=True)
            self.save_data(x_bounded=x_bounded,
                           jac_mean=jac_mean,
                           jac_var=jac_var,
                           adam_iter=adam_iter,
                           x_bounded_norm_ref=x_bounded_norm_ref)

            if adam_iter >= self.maxiter:
                #self.n_iter += 1
                break
                
            # Update Average Gradients
            jac_mean = beta_ADAM1*jac_mean + (1 - beta_ADAM1)*jac
            jac_var = beta_ADAM2*jac_var + (1 - beta_ADAM2)*jac**2
            
            # Unbias Average Gradients
            jac_mean_unbiased = jac_mean/(1 - beta_ADAM1**adam_iter)
            jac_var_unbiased = jac_var/(1 - beta_ADAM2**adam_iter)
            
            # Update Variables
            if self.n_iter == 0:
                x_bounded_norm_ref = 1
                x_bounded_norm = 1
            elif self.n_iter == 1:
                x_bounded_norm_ref = np.linalg.norm(x_bounded[:self.Ndim])
                x_bounded_norm = np.linalg.norm(x_bounded[:self.Ndim])
            else:
                x_bounded_norm = np.linalg.norm(x_bounded[:self.Ndim])

            #eta_sched = eta_ADAM.copy()
            eta_sched = eta_ADAM*(x_bounded_norm/x_bounded_norm_ref)**(1/3)
            if comm.rank == 0:
                print('  %6.4f  | %7.5f |   %5.2f   |' %(x_bounded_norm, eta_sched[0], t_rem), flush=True)
            x -= eta_sched*jac_mean_unbiased/(np.sqrt(jac_var_unbiased) + 1e-8)
            # eta *= 0.95
            
            self.x_bounded_norm_hist = np.append(self.x_bounded_norm_hist, x_bounded_norm)
            self.eta_sched_hist = np.append(self.eta_sched_hist, eta_sched[0])
            
            self.n_iter += 1

    def run(self, n_seed, output_filename, x0=None, eta_mu=0.01, eta_sigma=1.0, load_data=False):
        if comm.rank == 0:
            print('### Ensemble Optimization (seed = ' + str(n_seed) + ')\n', flush=True)
    
        np.random.seed(n_seed)
        self.output_filename = output_filename
        self.optimizer = optimizer
        
        sigma_ensemble = self.sigma_ensemble_max/2

        lb = np.zeros(self.Ndim+self.Nsigma)
        lb[:self.Ndim] = -1
        lb[self.Ndim:] = 0.99*self.sigma_ensemble_max/2
        ub = np.zeros(self.Ndim+self.Nsigma)
        ub[:self.Ndim] = 1
        ub[self.Ndim:] = 1.01*self.sigma_ensemble_max/2
        #ub[self.Ndim:] = 10.0
        
        if load_data:
            data_file1 = output_filename + "_GEGD_results.npz"
            data_file2 = output_filename + "_GEGD_density_hist.npz"
                
            with np.load(data_file1) as data:
                self.n_iter = data['n_iter']
                
                self.N_high_fidelity_hist = data['N_high_fidelity_hist'][:self.n_iter]
                self.N_low_fidelity_hist = data['N_low_fidelity_hist'][:self.n_iter]
                self.var_reduction_hist = data['var_reduction_hist'][:self.n_iter]
                self.N_eff_hist = data['N_eff_hist'][:self.n_iter]
                self.cost_ensemble_hist = data['cost_ensemble_hist'][:self.n_iter]
                self.cost_ensemble_ctrl_hist = data['cost_ensemble_ctrl_hist'][:self.n_iter]
                self.best_cost_hist = data['best_cost_hist'][:self.n_iter]
                self.cost_ensemble_sigma_hist = data['cost_ensemble_sigma_hist'][:self.n_iter]
                self.cost_ensemble_ctrl_sigma_hist = data['cost_ensemble_ctrl_sigma_hist'][:self.n_iter]
                self.ctrlVarCoeff_mu_hist = data['ctrlVarCoeff_mu_hist'][:self.n_iter]
                self.ctrlVarCoeff_sigma_hist = data['ctrlVarCoeff_sigma_hist'][:self.n_iter]
                self.corr_f_hist = data['corr_f_hist'][:self.n_iter]
                self.x_bounded_norm_hist = data['x_bounded_norm_hist'][:self.n_iter]
                self.eta_sched_hist = data['eta_sched_hist'][:self.n_iter]
                if self.Nsigma == 1:
                    self.sigma_ensemble_hist = data['sigma_ensemble_hist'][:self.n_iter]
                adam_iter = data['adam_iter'] - 1
                x_bounded_norm_ref = data['x_bounded_norm_ref']
                
            with np.load(data_file2) as data:
                self.x_latent_hist = data['x_latent_hist'][:self.n_iter,:]
                self.x_hist = data['x_hist'][:self.n_iter,:]
                self.best_x_hist = data['best_x_hist'][:self.n_iter,:]
                if self.Nsigma > 1:
                    self.sigma_ensemble_hist = data['sigma_ensemble_hist'][:self.n_iter,:]
                    self.sigma_fp_hist = data['sigma_fp_hist'][:self.n_iter,:]
                self.corr_mu_hist = data['corr_mu_hist'][:self.n_iter,:]
                self.corr_sigma_hist = data['corr_sigma_hist'][:self.n_iter,:]
                x0 = data['x_bounded']
                jac_mean = data['jac_mean']
                jac_var = data['jac_var']
            
        else:
            self.n_iter = 0
            
            self.N_high_fidelity_hist = np.zeros(0)
            self.N_low_fidelity_hist = np.zeros(0)
            self.var_reduction_hist = np.zeros(0)
            self.N_eff_hist = np.zeros(0)
            self.x_latent_hist = None
            self.x_hist = None
            self.best_x_hist = None
            self.cost_ensemble_hist = np.zeros(0)
            self.cost_ensemble_ctrl_hist = np.zeros(0)
            self.best_cost_hist = np.zeros(0)
            self.cost_ensemble_sigma_hist = np.zeros(0)
            self.cost_ensemble_ctrl_sigma_hist = np.zeros(0)
            if self.Nsigma == 1:
                self.sigma_ensemble_hist = np.zeros(0)
            else:
                self.sigma_ensemble_hist = None
                self.sigma_fp_hist = None
            self.ctrlVarCoeff_mu_hist = np.zeros(0)
            self.ctrlVarCoeff_sigma_hist = np.zeros(0)
            self.corr_mu_hist = None
            self.corr_sigma_hist = None
            self.corr_f_hist = np.zeros(0)
            self.x_bounded_norm_hist = np.zeros(0)
            self.eta_sched_hist = np.zeros(0)
            
            if x0 is None:
                # Initial Structure
                x0 = np.hstack((np.zeros(self.Ndim), sigma_ensemble*np.ones(self.Nsigma)))
            
            jac_mean = None
            jac_var = None
            adam_iter = None
            x_bounded_norm_ref = None
        
        if comm.rank == 0:
            if self.Nsigma == 1:
                print('    | Iter | N_acc | N_inacc | var_red | N_eff |  sigma  | CVCoeff | corr(f) | corr(g) |  Cost Ens  | Cost Ens CV | Cost Best |  x norm  | ADAM_LR | t_rem(hr) |',
                      flush=True)
            else:
                print('    | Iter | N_acc | N_inacc | var_red | N_eff | sig_min | sig_max | CVCoeff | corr(f) | corr(g) |  Cost Ens  | StD Ens | Cost Ens CV | StD Ens CV | Cost Best |  x norm  | ADAM_LR | t_rem(hr) |',
                      flush=True)

        eta_ADAM = np.ones(self.Ndim+self.Nsigma)
        eta_ADAM[:self.Ndim] = eta_mu
        eta_ADAM[self.Ndim:] = eta_sigma
        self.ADAM(x0, lb, ub, 0.9, 0.999, eta_ADAM,
                  jac_mean=jac_mean, jac_var=jac_var, adam_iter=adam_iter, x_bounded_norm_ref=x_bounded_norm_ref)
        
        self.save_data()
    
    def save_data(self, x_bounded=None, jac_mean=None, jac_var=None, adam_iter=None, x_bounded_norm_ref=None):
        if comm.rank == 0:
            if x_bounded is None:
                with np.load(output_filename + "_GEGD_results.npz") as data:
                    adam_iter = data['adam_iter']
                    x_bounded_norm_ref = data['x_bounded_norm_ref']
                
                with np.load(output_filename + "_GEGD_density_hist.npz") as data:
                    x_bounded = data['x_bounded']
                    jac_mean = data['jac_mean']
                    jac_var = data['jac_var']
        
            # Customize below
            if self.Nsigma == 1:
                np.savez(output_filename + "_GEGD_results",
                         N_high_fidelity_hist=self.N_high_fidelity_hist,
                         N_low_fidelity_hist=self.N_low_fidelity_hist,
                         var_reduction_hist=self.var_reduction_hist,
                         N_eff_hist=self.N_eff_hist,
                         cost_ensemble_hist=self.cost_ensemble_hist,
                         cost_ensemble_sigma_hist=self.cost_ensemble_sigma_hist,
                         cost_ensemble_ctrl_hist=self.cost_ensemble_ctrl_hist,
                         cost_ensemble_ctrl_sigma_hist=self.cost_ensemble_ctrl_sigma_hist,
                         best_cost_hist=self.best_cost_hist,
                         ctrlVarCoeff_mu_hist=self.ctrlVarCoeff_mu_hist,
                         ctrlVarCoeff_sigma_hist=self.ctrlVarCoeff_sigma_hist,
                         corr_f_hist=self.corr_f_hist,
                         x_bounded_norm_hist=self.x_bounded_norm_hist,
                         eta_sched_hist=self.eta_sched_hist,
                         sigma_ensemble_hist=self.sigma_ensemble_hist,
                         n_iter=self.n_iter,
                         adam_iter=adam_iter,
                         x_bounded_norm_ref=x_bounded_norm_ref,
                         )
                         
                np.savez(output_filename + "_GEGD_density_hist",
                         x_hist=self.x_hist,
                         x_latent_hist=self.x_latent_hist,
                         best_x_hist=self.best_x_hist,
                         corr_mu_hist=self.corr_mu_hist,
                         corr_sigma_hist=self.corr_sigma_hist,
                         x_bounded=x_bounded,
                         jac_mean=jac_mean,
                         jac_var=jac_var,
                         )
            
            else:
                np.savez(output_filename + "_GEGD_results",
                         N_high_fidelity_hist=self.N_high_fidelity_hist,
                         N_low_fidelity_hist=self.N_low_fidelity_hist,
                         var_reduction_hist=self.var_reduction_hist,
                         N_eff_hist=self.N_eff_hist,
                         cost_ensemble_hist=self.cost_ensemble_hist,
                         cost_ensemble_sigma_hist=self.cost_ensemble_sigma_hist,
                         cost_ensemble_ctrl_hist=self.cost_ensemble_ctrl_hist,
                         cost_ensemble_ctrl_sigma_hist=self.cost_ensemble_ctrl_sigma_hist,
                         best_cost_hist=self.best_cost_hist,
                         ctrlVarCoeff_mu_hist=self.ctrlVarCoeff_mu_hist,
                         ctrlVarCoeff_sigma_hist=self.ctrlVarCoeff_sigma_hist,
                         corr_f_hist=self.corr_f_hist,
                         x_bounded_norm_hist=self.x_bounded_norm_hist,
                         eta_sched_hist=self.eta_sched_hist,
                         n_iter=self.n_iter,
                         adam_iter=adam_iter,
                         x_bounded_norm_ref=x_bounded_norm_ref,
                         )
                         
                np.savez(output_filename + "_GEGD_density_hist",
                         x_hist=self.x_hist,
                         sigma_ensemble_hist=self.sigma_ensemble_hist,
                         sigma_fp_hist=self.sigma_fp_hist,
                         x_latent_hist=self.x_latent_hist,
                         best_x_hist=self.best_x_hist,
                         corr_mu_hist=self.corr_mu_hist,
                         corr_sigma_hist=self.corr_sigma_hist,
                         x_bounded=x_bounded,
                         jac_mean=jac_mean,
                         jac_var=jac_var,
                         )

    def sample_cost_fct_landscape(self, output_filename, Nensemble, r_CV, dx1, dx2, Ngrid1interp, Ngrid2interp, Ngrid1, Ngrid2, read_save_data=False, zoom=1):
        if comm.rank == 0:
            print('### Sampling Cost Function Landscape', flush=True)
        
        if read_save_data:
            with np.load(output_filename + "_GEGD_landscape.npz") as data:
                cost_STE = data['cost_STE']
                x_STE = data['x_STE']
                jac_STE = data['jac_STE']
                cost_ensemble = data['cost_ensemble']
                jac_ensemble_mu = data['jac_ensemble_mu']
                jac_NES_mu = data['jac_NES_mu']
                jac_ensemble_sigma = data['jac_ensemble_sigma']
                jac_NES_sigma = data['jac_NES_sigma']
            n1_start = np.min(np.argwhere(cost_ensemble[:,0]==0))
            
        else:
            cost_STE = np.zeros((Ngrid1, Ngrid2))
            x_STE = np.zeros((Ngrid1, Ngrid2, self.Nx, self.Ny))
            jac_STE = np.zeros((Ngrid1, Ngrid2, self.Ndim))
            cost_ensemble = np.zeros((Ngrid1, Ngrid2))
            jac_ensemble_mu = np.zeros((Ngrid1, Ngrid2, self.Ndim))
            jac_ensemble_sigma = np.zeros((Ngrid1, Ngrid2, self.Nsigma))
            jac_NES_mu = np.zeros((Ngrid1, Ngrid2, self.Ndim))
            jac_NES_sigma = np.zeros((Ngrid1, Ngrid2, self.Nsigma))
            n1_start = 0
        
        lb = np.zeros(self.Ndim+self.Nsigma)
        lb[:self.Ndim] = -1
        lb[self.Ndim:] = 0.99*self.sigma_ensemble_max/2
        ub = np.zeros(self.Ndim+self.Nsigma)
        ub[:self.Ndim] = 1
        ub[self.Ndim:] = 1.01*self.sigma_ensemble_max/2
        
        sigma_ensemble = self.sigma_ensemble_max/2
        
        if self.Nsigma == 1:
            sigma_fp = sigma_ensemble.copy()
        else:
            sigma_scaled = 2*(sigma_ensemble - lb[self.Ndim:])/(ub[self.Ndim:] - lb[self.Ndim:]) - 1
            sigma_fp_scaled = dtf.filter_and_project(sigma_scaled, self.symmetry, self.periodic, self.Nx, self.Ny, self.brush_size, self.sigma_filter, self.beta_proj_sigma, padding=self.padding)
            sigma_fp = (sigma_fp_scaled + 1)*(ub[self.Ndim:] - lb[self.Ndim:])/2 + lb[self.Ndim:]
        cov, cov_inv = self.construct_cov_matrix(sigma_fp)

        if comm.rank == 0:
            np.savez(output_filename + "_GEGD_landscape", dx1=dx1, dx2=dx2)

        self.Nensemble = Nensemble
        self.r_CV = r_CV
        for n1 in range(n1_start, Ngrid1):
            if comm.rank == 0:
                print('\tN1: ' + str(n1) + ' | N2: ', end='', flush=True)
            rn = n1/(Ngrid1 - 1)/zoom
                
            for n2 in range(Ngrid2):
                if comm.rank == 0:
                    print(str(n2) + ' ', end='', flush=True)
                
                if n2 == Ngrid2 - 1:
                    cost_STE[n1,n2] = cost_STE[n1,0]
                    x_interp = np.linspace(0, 1, Ngrid1interp)
                    y_interp = np.linspace(0, 1, Ngrid2interp)
                    cost_STE_interp = interpn((np.linspace(0, 1, Ngrid1), np.linspace(0, 1, Ngrid2)),
                                              cost_STE,
                                              np.array(list(product(x_interp, y_interp))),
                                              method='nearest').reshape(Ngrid1interp, Ngrid2interp)
                    x_STE[n1,n2,:,:] = x_STE[n1,0,:,:]
                    jac_STE[n1,n2,:] = jac_STE[n1,0,:]
                    
                    cost_ensemble[n1,n2] = cost_ensemble[n1,0]
                    cost_ensemble_interp = interpn((np.linspace(0, 1, Ngrid1), np.linspace(0, 1, Ngrid2)),
                                                   cost_ensemble,
                                                   np.array(list(product(x_interp, y_interp))),
                                                   method='linear').reshape(Ngrid1interp, Ngrid2interp)
                    jac_ensemble_mu[n1,n2,:] = jac_ensemble_mu[n1,0,:]
                    jac_ensemble_sigma[n1,n2,:] = jac_ensemble_sigma[n1,0,:]
                    jac_NES_mu[n1,n2,:] = jac_NES_mu[n1,0,:]
                    jac_NES_sigma[n1,n2,:] = jac_NES_sigma[n1,0,:]
                
                else:
                    th_n = 2*np.pi/(Ngrid2 - 1)*n2
                    xn = rn*(np.cos(th_n)*dx1 + np.sin(th_n)*dx2)
                
                    dx = np.random.multivariate_normal(np.zeros(self.Ndim), cov, size=self.r_CV*self.Nensemble)
                    cost_all, density_all, grad_STE, grad_ensemble = self.ensemble_jacobian(xn, dx, sigma_fp, True)
                        
                    cost_STE[n1,n2] = cost_all[0]
                    x_interp = np.linspace(0, 1, Ngrid1interp)
                    y_interp = np.linspace(0, 1, Ngrid2interp)
                    cost_STE_interp = interpn((np.linspace(0, 1, Ngrid1), np.linspace(0, 1, Ngrid2)),
                                              cost_STE,
                                              np.array(list(product(x_interp, y_interp))),
                                              method='nearest').reshape(Ngrid1interp, Ngrid2interp)
                    x_STE[n1,n2,:,:] = density_all[1].reshape(self.Nx, self.Ny)
                    jac_STE[n1,n2,:] = grad_STE[1]
                    
                    cost_ensemble[n1,n2] = cost_all[1]
                    cost_ensemble_interp = interpn((np.linspace(0, 1, Ngrid1), np.linspace(0, 1, Ngrid2)),
                                                   cost_ensemble,
                                                   np.array(list(product(x_interp, y_interp))),
                                                   method='linear').reshape(Ngrid1interp, Ngrid2interp)
                    jac_ensemble_mu[n1,n2,:] = grad_ensemble[1][:self.Ndim]
                    jac_ensemble_sigma[n1,n2,:] = grad_ensemble[1][self.Ndim:]
                    jac_NES_mu[n1,n2,:] = cov @ jac_ensemble_mu[n1,n2,:]
                    jac_NES_sigma[n1,n2,:] = sigma_fp**2*jac_ensemble_sigma[n1,n2,:]/2
                    jac_NES_sigma[n1,n2,:] *= 2/(ub[self.Ndim:] - lb[self.Ndim:]) # accounts for covariance rescaling
            
            if comm.rank == 0:
                print('', flush=True)
                
                np.savez(output_filename + "_GEGD_landscape",
                         dx1=dx1,
                         dx2=dx2,
                         cost_STE=cost_STE,
                         cost_STE_interp=cost_STE_interp,
                         x_STE=x_STE,
                         jac_STE=jac_STE,
                         cost_ensemble=cost_ensemble,
                         cost_ensemble_interp=cost_ensemble_interp,
                         jac_ensemble_mu=jac_ensemble_mu,
                         jac_ensemble_sigma=jac_ensemble_sigma,
                         jac_NES_mu=jac_NES_mu,
                         jac_NES_sigma=jac_NES_sigma)