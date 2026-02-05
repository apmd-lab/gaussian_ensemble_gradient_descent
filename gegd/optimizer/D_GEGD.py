import numpy as np
import torch
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
                 min_feature_size,
                 sigma_RBF,
                 perturbation_rank=1,
                 var_ensemble=1.0,
                 asymmetry_factor=0.5,
                 upsample_ratio=1,
                 beta_proj=8,
                 feasible_design_generation_method='brush',
                 brush_shape='circle',
                 coeff_exp=5,
                 cost_threshold=0,
                 cost_obj=None,
                 Nthreads=1,
                 verbosity=1,
                 cuda_ind=0,
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
        self.feasible_design_generation_method = feasible_design_generation_method
        self.min_feature_size = min_feature_size
        self.perturbation_rank = perturbation_rank
        self.var_ensemble = var_ensemble
        self.asymmetry_factor = asymmetry_factor
        self.upsample_ratio = upsample_ratio
        self.brush_shape = brush_shape
        self.sigma_filter = sigma_RBF #min_feature_size/2/np.sqrt(2)
        self.sigma_RBF = sigma_RBF #min_feature_size/2/np.sqrt(2)
        self.high_fidelity_setting = high_fidelity_setting
        self.low_fidelity_setting = low_fidelity_setting
        self.coeff_exp = coeff_exp
        self.cost_threshold = cost_threshold
        self.cost_obj = cost_obj
        self.Nthreads = Nthreads
        self.verbosity = verbosity
        self.cuda_ind = cuda_ind

        self.device = torch.device(f'cuda:{cuda_ind}') if torch.cuda.is_available() else torch.device('cpu')
        
        # Get Number of Independent Parameters
        if symmetry == 0:
            self.Ndim = Nx*Ny
           
        elif symmetry == 1:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*Ny)
        
        elif symmetry == 2:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*np.floor(Ny/2 + 0.5))
        
        elif symmetry == 4:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*(np.floor(Nx/2 + 0.5) + 1)/2)

        t1 = time.time()
        self.construct_gaussian_covariance()
        t2 = time.time()
        if comm.rank == 0 and self.verbosity >= 2:
            print('--> Gaussian Covariance Construction: ', t2-t1, flush=True)
    
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
        
        self.cov_g = torch.tensor(self.cov_g + eps*np.identity(self.Ndim), dtype=torch.float64, device=self.device)
    
    def construct_cov_matrix(self, s):
        s = s.reshape(self.Ndim, self.perturbation_rank, order='F')
        s = torch.tensor(s, dtype=torch.float64, device=self.device)

        cov = self.var_ensemble*(self.cov_g + s @ s.T)
        cov_inv = torch.linalg.inv(cov)
        
        return cov, cov_inv
    
    def s_derivative(self, s, dx, cov_inv, p):
        s = s.reshape(self.Ndim, self.perturbation_rank, order='F')
        s = torch.tensor(s, dtype=torch.float64, device=self.device)
        dx = torch.tensor(dx, dtype=torch.float64, device=self.device)
        dx = dx.unsqueeze(-1)

        z = cov_inv @ dx
        grad = ((z @ z.T - cov_inv) @ s) * p + self.asymmetry_factor * z

        return np.squeeze(grad.detach().cpu().numpy())
    
    def ensemble_jacobian(self, s, dx):
        cov, cov_inv = self.construct_cov_matrix(s)
        
        # Get Brush Binarized Densities ------------------------------------------------------------
        t1 = time.time()
        x_sample = dtf.binarize(
            np.zeros(self.Ndim),
            self.symmetry,
            self.periodic,
            self.Nx,
            self.Ny,
            self.min_feature_size,
            self.brush_shape,
            self.beta_proj,
            self.sigma_filter,
            dx=dx,
            upsample_ratio=self.upsample_ratio,
            padding=self.padding,
            method=self.feasible_design_generation_method,
            Nthreads=self.Nthreads,
            cuda_ind=self.cuda_ind,
            ).astype(np.float64)
        t2 = time.time()
        if comm.rank == 0 and self.verbosity >= 2:
            print('--> Feasible Design Generation: ', t2-t1, flush=True)
        
        # Sample Modified Cost Function --------------------------------------------------------------
        self.cost_obj.set_accuracy(self.high_fidelity_setting)

        t1 = time.time()
        f = np.zeros(self.Nensemble)
        f_logDeriv = np.zeros((self.Nensemble, self.Ndim, self.perturbation_rank))
        for n in range(self.Nensemble):
            f_temp = self.cost_obj.get_cost(x_sample[n,:], False)
            f_shifted = (f_temp - self.cost_threshold)/(self.cost_threshold + 1)
            f[n] = -np.exp(-self.coeff_exp*f_shifted)
            f_logDeriv[n,:,:] = self.s_derivative(s, dx[n,:], cov_inv, f[n])
        t2 = time.time()
        if comm.rank == 0 and self.verbosity >= 2:
            print('--> Cost Computation: ', t2-t1, flush=True)
        
        # Get Best Cost -----------------------------------------------------------------------------------
        f_best = (-np.log(-np.min(f))/self.coeff_exp)*(self.cost_threshold + 1) + self.cost_threshold
        x_best = x_sample[np.argmin(f),:].copy()
        
        # Sample Control Variate Function ----------------------------------------------------------------
        self.cost_obj.set_accuracy(self.low_fidelity_setting)
        
        f_ctrl_all = np.zeros(self.r_CV*self.Nensemble)
        f_ctrl_logDeriv_all = np.zeros((self.r_CV*self.Nensemble, self.Ndim, self.perturbation_rank))
        for n in range(self.r_CV*self.Nensemble):
            f_temp = self.cost_obj.get_cost(x_sample[n,:], False)
            f_shifted = (f_temp - self.cost_threshold)/(self.cost_threshold + 1)
            f_ctrl_all[n] = -np.exp(-self.coeff_exp*f_shifted)
            f_ctrl_logDeriv_all[n,:,:] = self.s_derivative(s, dx[n,:], cov_inv, f_ctrl_all[n])
        
        # Expectation of the Control Variate Function -----------------------------------------------------
        ctrlVarExp_f = np.mean(f_ctrl_all)
        ctrlVarExp_s = np.mean(f_ctrl_logDeriv_all, axis=0, keepdims=True)

        # Control Variate Coefficient (expectation) --------------------------------------------------------------------
        f_ctrl = f_ctrl_all[:self.Nensemble]
        mean_f = np.mean(f)
        mean_f_ctrl = np.mean(f_ctrl)
        
        cov_f_f_ctrl = (1/(self.Nensemble - 1))*np.sum((f - mean_f)*(f_ctrl - mean_f_ctrl))
        var_f = (1/(self.Nensemble - 1))*np.sum((f - mean_f)**2)
        var_ctrl = (1/(self.Nensemble - 1))*np.sum((f_ctrl - mean_f_ctrl)**2)

        ctrlVarCoeff_f = cov_f_f_ctrl/var_ctrl
        corr_f = cov_f_f_ctrl/np.sqrt(var_f*var_ctrl)
        
        # Control Variate Coefficient (expectation gradient wrt s) --------------------------------------------------------------------
        f_s = f_logDeriv.copy()
        f_ctrl_s = f_ctrl_logDeriv_all[:self.Nensemble,:,:]
        mean_f = np.mean(f_s, axis=0)
        mean_ctrl = np.mean(f_ctrl_s, axis=0)
        
        cov_f_ctrl = (1/(self.Nensemble - 1))*np.sum((f_s - mean_f)*(f_ctrl_s - mean_ctrl), axis=0)
        var_f = (1/(self.Nensemble - 1))*np.sum((f_s - mean_f)**2, axis=0)
        var_ctrl = (1/(self.Nensemble - 1))*np.sum((f_ctrl_s - mean_ctrl)**2, axis=0)

        weights = np.ones(self.Ndim*self.perturbation_rank) #1/np.hstack((sigma_ensemble, sigma_ensemble))**2
        ctrlVarCoeff_s = np.average(cov_f_ctrl.reshape(-1), weights=weights)/np.average(var_ctrl.reshape(-1), weights=weights)
        corr_s = (cov_f_ctrl/np.sqrt(var_f*var_ctrl)).reshape(-1, order='F')

        # Control Variate Estimation of the expectation --------------------------------------------------------------
        if self.r_CV == 1:
            f_est = f.copy()
        else:
            f_est = f - ctrlVarCoeff_f*(f_ctrl - ctrlVarExp_f)
        
        # Control Variate Estimation of the gradient --------------------------------------------------------------
        jac_fp_ensemble = f_logDeriv.copy()
        if self.r_CV > 1:
            jac_fp_ensemble -= ctrlVarCoeff_s*(f_ctrl_s - ctrlVarExp_s)

        jac_fp_ensemble = np.mean(jac_fp_ensemble, axis=0)

        s_reshape = s.reshape(self.Ndim, self.perturbation_rank)
        jac_latent_ensemble = np.zeros((self.Ndim, self.perturbation_rank))
        for r in range(self.perturbation_rank):
            jac_fp_ensemble_sym = symOp.symmetrize_jacobian(jac_fp_ensemble[:,r], self.symmetry, self.Nx, self.Ny)
            jac_latent_ensemble[:,r] = dtf.backprop_filter_and_project(jac_fp_ensemble_sym, s_reshape[:,r], self.symmetry, self.periodic, self.Nx, self.Ny, self.min_feature_size, self.sigma_filter, 0, padding=self.padding)
        
        jac_latent_ensemble = jac_latent_ensemble.reshape(-1, order='F')

        return np.mean(f), np.std(f), np.mean(f_ctrl_all), np.std(f_ctrl_all), jac_latent_ensemble, ctrlVarCoeff_s, corr_f, corr_s, f_best, x_best
    
    def ADAM(self,
             s,
             beta_ADAM1,
             beta_ADAM2,
             eta_ADAM,
             jac_mean=None,
             jac_var=None,
             adam_iter=None,
             ):
    
        if comm.rank == 0 and self.verbosity >= 2:
            print('--> Starting ADAM', flush=True)

        if jac_mean is None:
            jac_mean = np.zeros_like(s)
        if jac_var is None:
            jac_var = np.zeros_like(s)
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
            
            if self.corr_s_hist is None:
                corr_s = 0
            elif self.corr_s_hist.ndim == 1:
                corr_s = np.mean(self.corr_s_hist)
            else:
                corr_s = np.mean(self.corr_s_hist[-1,:])
            
            var_reduction = np.zeros(N.size)
            var_reduction[r_CV>0] = (1 - ((r_CV[r_CV>0] - 1)/r_CV[r_CV>0])*corr_s**2)/N[r_CV>0]
            var_reduction[r_CV<=0] = 1/N[r_CV<=0]
            ind_opt = np.nanargmin(var_reduction)
            self.Nensemble = N[ind_opt]
            self.r_CV = r_CV[ind_opt]
            
            var_reduction = (1 - ((self.r_CV - 1)/self.r_CV)*corr_s**2)/self.Nensemble
            
            t1 = time.time()
            s_reshape = s.reshape(self.Ndim, self.perturbation_rank, order='F')
            s_filtered = np.zeros((self.Ndim, self.perturbation_rank))
            for r in range(self.perturbation_rank):
                s_filtered[:,r] = dtf.filter_and_project(s_reshape[:,r], self.symmetry, self.periodic, self.Nx, self.Ny, self.min_feature_size, self.sigma_filter, 0, padding=self.padding)
            mu = torch.tensor(self.asymmetry_factor * np.sum(s_filtered, axis=1), dtype=torch.float64, device=self.device).unsqueeze(0)
            s_filtered = s_filtered.reshape(-1, order='F')
            cov, cov_inv = self.construct_cov_matrix(s_filtered)
            L = torch.linalg.cholesky(cov)
            lam = torch.linalg.eigvalsh(cov)
            lam = lam.detach().cpu().numpy()

            z = np.random.normal(size=(self.r_CV * self.Nensemble, self.Ndim))
            z = torch.tensor(z, dtype=torch.float64, device=self.device)
            dx = mu + z @ L.T
            dx = dx.detach().cpu().numpy()
            t2 = time.time()
            if comm.rank == 0 and self.verbosity >= 2:
                print('--> Random Sample Generation: ', t2-t1, flush=True)

            t1 = time.time()
            loss_mean, loss_std, loss_ctrl_mean, loss_ctrl_std, jac, ctrlVarCoeff_s, corr_f, corr_s, f_best, x_best = self.ensemble_jacobian(s_filtered, dx)
            t2 = time.time()
            if comm.rank == 0 and self.verbosity >= 2:
                print('--> Ensemble Jacobian Computation: ', t2-t1, flush=True)
            
            self.N_high_fidelity_hist = np.append(self.N_high_fidelity_hist, self.Nensemble)
            self.N_low_fidelity_hist = np.append(self.N_low_fidelity_hist, self.r_CV*self.Nensemble)
            self.var_reduction_hist = np.append(self.var_reduction_hist, var_reduction)
            self.N_eff_hist = np.append(self.N_eff_hist, int(1/var_reduction))
            
            if self.s_hist is None:
                self.s_hist = s_filtered.copy()
            else:
                self.s_hist = np.vstack((self.s_hist, s_filtered))
            
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
            
            self.ctrlVarCoeff_s_hist = np.append(self.ctrlVarCoeff_s_hist, ctrlVarCoeff_s)
            self.corr_f_hist = np.append(self.corr_f_hist, corr_f)
            
            if self.corr_s_hist is None:
                self.corr_s_hist = corr_s.copy()
            else:
                self.corr_s_hist = np.vstack((self.corr_s_hist, corr_s))

            self.condition_number_hist = np.append(self.condition_number_hist, np.max(lam)/np.min(lam))
        
            t2 = time.time()
            t_rem = (t2 - t1)*(self.maxiter - self.n_iter + 1)/3600
            
            if comm.rank == 0:
                print('    | %4d |  %4d |   %4d  | %7.5f |  %5.2f  | %7.4f | %7.4f |  %9.2f |  %9.2f  |  %8.3f | %16.1f |' %(
                    self.n_iter,
                    self.Nensemble,
                    self.r_CV*self.Nensemble,
                    var_reduction,
                    ctrlVarCoeff_s,
                    np.mean(corr_f),
                    np.mean(corr_s),
                    -np.log(-loss_mean)/self.coeff_exp,
                    -np.log(-loss_ctrl_mean)/self.coeff_exp,
                    self.best_cost_hist[-1],
                    np.log10(self.condition_number_hist[-1]),
                    ), end='', flush=True)
            
            t1 = time.time()
            self.save_data(
                s=s,
                cov=cov.detach().cpu().numpy(),
                jac_mean=jac_mean,
                jac_var=jac_var,
                adam_iter=adam_iter,
                )
            t2 = time.time()
            if comm.rank == 0 and self.verbosity >= 2:
                print('--> Data Saving: ', t2-t1, flush=True)
            
            if adam_iter >= self.maxiter:
                #self.n_iter += 1
                break
                
            # Update Average Gradients
            jac_mean = beta_ADAM1*jac_mean + (1 - beta_ADAM1)*jac
            jac_var = beta_ADAM2*jac_var + (1 - beta_ADAM2)*jac**2
            
            # Unbias Average Gradients
            jac_mean_unbiased = jac_mean/(1 - beta_ADAM1**adam_iter)
            jac_var_unbiased = jac_var/(1 - beta_ADAM2**adam_iter)

            if comm.rank == 0:
                print('   %5.2f   |' %t_rem, flush=True)

            s -= eta_ADAM*jac_mean_unbiased/(np.sqrt(jac_var_unbiased) + 1e-8)
            
            self.n_iter += 1

    def run(self, n_seed, output_filename, s0=None, s_mag0=0.01, eta=0.01, load_data=False):
        if comm.rank == 0 and self.verbosity >= 1:
            print('### Directional GEGD (seed = ' + str(n_seed) + ')\n', flush=True)
    
        np.random.seed(n_seed)
        self.output_filename = output_filename
        
        if load_data:
            data_file1 = output_filename + "_D_GEGD_results.npz"
            data_file2 = output_filename + "_D_GEGD_density_hist.npz"
                
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
                self.ctrlVarCoeff_s_hist = data['ctrlVarCoeff_s_hist'][:self.n_iter]
                self.corr_f_hist = data['corr_f_hist'][:self.n_iter]
                self.condition_number_hist = data['condition_number_hist'][:self.n_iter]
                adam_iter = data['adam_iter'] - 1
                
            with np.load(data_file2) as data:
                self.s_hist = data['s_hist'][:self.n_iter,:]
                self.best_x_hist = data['best_x_hist'][:self.n_iter,:]
                self.corr_s_hist = data['corr_s_hist'][:self.n_iter,:]
                s0 = data['s']
                jac_mean = data['jac_mean']
                jac_var = data['jac_var']
            
        else:
            self.n_iter = 0
            
            self.N_high_fidelity_hist = np.zeros(0)
            self.N_low_fidelity_hist = np.zeros(0)
            self.var_reduction_hist = np.zeros(0)
            self.N_eff_hist = np.zeros(0)
            self.s_hist = None
            self.best_x_hist = None
            self.cost_ensemble_hist = np.zeros(0)
            self.cost_ensemble_ctrl_hist = np.zeros(0)
            self.best_cost_hist = np.zeros(0)
            self.cost_ensemble_sigma_hist = np.zeros(0)
            self.cost_ensemble_ctrl_sigma_hist = np.zeros(0)
            self.ctrlVarCoeff_s_hist = np.zeros(0)
            self.corr_s_hist = None
            self.corr_f_hist = np.zeros(0)
            self.condition_number_hist = np.zeros(0)
            
            if s0 is None:
                # Initial Structure
                s0 = np.random.normal(0, s_mag0, size=self.Ndim*self.perturbation_rank)
            
            jac_mean = None
            jac_var = None
            adam_iter = None
        
        if comm.rank == 0 and self.verbosity >= 1:
            print('    | Iter | N_acc | N_inacc | var_red | CVCoeff | corr(f) | corr(g) |  Cost Ens  | Cost Ens CV | Cost Best | Cond Num (log10) | t_rem(hr) |',
                flush=True)

        self.ADAM(
            s0,
            0.9,
            0.999,
            eta,
            jac_mean=jac_mean,
            jac_var=jac_var,
            adam_iter=adam_iter,
            )
        
        if comm.rank == 0 and self.verbosity >= 2:
            print('--> Saving Final Data', flush=True)
        self.save_data()
    
    def save_data(self, s=None, cov=None, jac_mean=None, jac_var=None, adam_iter=None):
        if comm.rank == 0:
            if s is None:
                with np.load(self.output_filename + "_D_GEGD_results.npz") as data:
                    adam_iter = data['adam_iter']
                
                with np.load(self.output_filename + "_D_GEGD_density_hist.npz") as data:
                    s = data['s']
                    cov = data['cov']
                    jac_mean = data['jac_mean']
                    jac_var = data['jac_var']
        
            if self.best_x_hist.ndim == 1:
                best_x_final = self.best_x_hist.copy()
                s_final = self.s_hist.copy()
            else:
                best_x_final = self.best_x_hist[-1,:]
                s_final = self.s_hist[-1,:]

            # Customize below
            np.savez(self.output_filename + "_D_GEGD_results",
                N_high_fidelity_hist=self.N_high_fidelity_hist,
                N_low_fidelity_hist=self.N_low_fidelity_hist,
                var_reduction_hist=self.var_reduction_hist,
                N_eff_hist=self.N_eff_hist,
                cost_ensemble_hist=self.cost_ensemble_hist,
                cost_ensemble_sigma_hist=self.cost_ensemble_sigma_hist,
                cost_ensemble_ctrl_hist=self.cost_ensemble_ctrl_hist,
                cost_ensemble_ctrl_sigma_hist=self.cost_ensemble_ctrl_sigma_hist,
                best_cost_hist=self.best_cost_hist,
                ctrlVarCoeff_s_hist=self.ctrlVarCoeff_s_hist,
                condition_number_hist=self.condition_number_hist,
                corr_f_hist=self.corr_f_hist,
                best_x_final=best_x_final,
                s_final=s_final,
                n_iter=self.n_iter,
                adam_iter=adam_iter,
                )
                        
            np.savez(self.output_filename + "_D_GEGD_density_hist",
                s_hist=self.s_hist,
                best_x_hist=self.best_x_hist,
                corr_s_hist=self.corr_s_hist,
                s=s,
                cov=cov,
                jac_mean=jac_mean,
                jac_var=jac_var,
                )