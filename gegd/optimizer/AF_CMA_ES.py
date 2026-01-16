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
                 Nsample,
                 symmetry,
                 periodic,
                 padding,
                 maxiter,
                 high_fidelity_setting,
                 min_feature_size,
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

    def CMA_ES(
        self,
        xmean=None,
        sigma=None,
        Cov=None,
        p_c=None,
        p_sigma=None,
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

        # Set Adaptation Parameters
        c_c = (4 + mu_eff / self.Ndim) / (self.Ndim + 4 + 2 * mu_eff / self.Ndim)
        c_sigma = (mu_eff + 2) / (self.Ndim + mu_eff + 5)
        c_1 = 2 / ((self.Ndim + 1.3)**2 + mu_eff)
        c_mu = np.min((1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.Ndim + 2)**2 + 2 * mu_eff / 2)))
        d_sigma = 1 + 2 * np.max((0, np.sqrt((mu_eff - 1) / (self.Ndim + 1)) - 1)) + c_sigma

        # Initialize Dynamic Internal Parameters
        p_c = p_c if p_c is not None else np.zeros(self.Ndim)
        p_sigma = p_sigma if p_sigma is not None else np.zeros(self.Ndim)
        B = B if B is not None else np.identity(self.Ndim)
        D = D if D is not None else np.identity(self.Ndim)
        xmean = xmean if xmean is not None else np.zeros(self.Ndim)
        sigma = sigma if sigma is not None else 0.5
        Cov = Cov if Cov is not None else np.identity(self.Ndim)
        eig_eval = eig_eval if eig_eval is not None else 0
        chi_N = self.Ndim**0.5 * (1 - 1 / (4 * self.Ndim) + 1 / (21 * self.Ndim**2))

        counteval = counteval if counteval is not None else 0

        while True:
            t1 = time.time()

            # Sample Candidates
            arz = np.random.normal(size=(self.Ndim, self.Nsample))
            arx = xmean[:,np.newaxis] + sigma * (B @ D @ arz)
            
            # Cost Evaluation
            cost, x_bin = self.get_cost_samples(arx.T)
            sorted_index = np.argsort(cost)
            cost = cost[sorted_index]
            arx = arx[:,sorted_index]
            x_bin = x_bin[sorted_index,:]
            counteval += self.Nsample

            if self.x_mean_hist is None:
                self.x_mean_hist = xmean.copy()
            else:
                self.x_mean_hist = np.vstack((self.x_mean_hist, xmean))

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
            self.sigma_hist = np.append(self.sigma_hist, sigma)
            
            t2 = time.time()
            t_rem = (t2 - t1)*(self.maxiter - self.n_iter + 1)/3600
            
            if comm.rank == 0:
                print('    | %4d |  %4d | %7.5f |  %9.2f |  %8.3f |   %5.2f   |' %(
                    self.n_iter,
                    self.Nsample,
                    sigma,
                    np.mean(cost),
                    self.best_cost_hist[-1],
                    t_rem,
                ), flush=True)

            self.save_data(
                p_c=p_c,
                p_sigma=p_sigma,
                B=B,
                D=D,
                xmean=xmean,
                sigma=sigma,
                Cov=Cov,
                eig_eval=eig_eval,
                counteval=counteval,
            )

            if self.n_iter > self.maxiter:
                self.n_iter += 1
                break

            # Update Mean
            xmean = arx[:,:mu] @ weights
            zmean = arz[:,:mu] @ weights

            # Update Evolution Path and Step Size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ zmean)
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * counteval / self.Nsample)) / chi_N < 1.4 + 2 / (self.Ndim + 1)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ D @ zmean)

            # Update Covariance Matrix
            Cov = (1 - c_1 - c_mu) * Cov \
                + c_1 * (p_c @ p_c.T + (1 - h_sigma) * c_c * (2 - c_c) * Cov) \
                + c_mu * (B @ D @ arz[:,:mu]) @ np.diag(weights) @ (B @ D @ arz[:,:mu]).T
            
            # Update Step Size
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_N - 1))

            # Update B and D
            if counteval - eig_eval > self.Nsample / ((c_1 + c_mu) * self.Ndim * 10):
                eig_eval = counteval
                Cov = np.triu(Cov) + np.triu(Cov, k=1).T
                [D, B] = np.linalg.eigh(Cov)
                D = np.sqrt(np.diag(D))
            
            self.n_iter += 1

    def run(self, n_seed, output_filename, x0=None, load_data=False):
        if comm.rank == 0:
            print('### CMA-ES (seed = ' + str(n_seed) + ')\n', flush=True)
    
        np.random.seed(n_seed)
        self.output_filename = output_filename
        
        if load_data:
            data_file1 = output_filename + "_CMA_ES_results.npz"
            data_file2 = output_filename + "_CMA_ES_density_hist.npz"
                
            with np.load(data_file1) as data:
                self.n_iter = data['n_iter']
                
                self.cost_ensemble_hist = data['cost_ensemble_hist'][:self.n_iter]
                self.best_cost_hist = data['best_cost_hist'][:self.n_iter]
                self.cost_ensemble_sigma_hist = data['cost_ensemble_sigma_hist'][:self.n_iter]
                self.sigma_hist = data['sigma_hist'][:self.n_iter]

                sigma = data['sigma']
                eig_eval = data['eig_eval']
                counteval = data['counteval']
                
            with np.load(data_file2) as data:
                self.x_latent_hist = data['x_latent_hist'][:self.n_iter,:]
                self.x_mean_hist = data['x_mean_hist'][:self.n_iter,:]
                self.best_x_hist = data['best_x_hist'][:self.n_iter,:]
                
                p_c = data['p_c']
                p_sigma = data['p_sigma']
                B = data['B']
                D = data['D']
                x0 = data['xmean']
                Cov = data['Cov']

        else:
            self.n_iter = 0
            
            self.x_latent_hist = None
            self.x_mean_hist = None
            self.best_x_hist = None
            self.cost_ensemble_hist = np.zeros(0)
            self.best_cost_hist = np.zeros(0)
            self.cost_ensemble_sigma_hist = np.zeros(0)
            self.sigma_hist = np.zeros(0)
            
            if x0 is None:
                # Initial Structure
                x0 = np.zeros(self.Ndim)
            sigma = None
            Cov = None
            p_c = None
            p_sigma = None
            B = None
            D = None
            eig_eval = None
            counteval = None
        
        if comm.rank == 0:
            print('    | Iter |   N   |  sigma  |  Cost Ens  | Cost Best | t_rem(hr) |',
                  flush=True)

        self.CMA_ES(
            xmean=x0,
            sigma=sigma,
            Cov=Cov,
            p_c=p_c,
            p_sigma=p_sigma,
            B=B,
            D=D,
            eig_eval=eig_eval,
            counteval=counteval,
        )
        
        self.save_data()
    
    def save_data(
        self,
        p_c=None,
        p_sigma=None,
        B=None,
        D=None,
        xmean=None,
        sigma=None,
        Cov=None,
        eig_eval=None,
        counteval=None,
    ):

        if comm.rank == 0:
            if p_c is None:
                with np.load(self.output_filename + "_CMA_ES_density_hist.npz") as data:
                    p_c = data['p_c']
                    p_sigma = data['p_sigma']
                    B = data['B']
                    D = data['D']
                    xmean = data['xmean']
                    Cov = data['Cov']

            if sigma is None:
                with np.load(self.output_filename + "_CMA_ES_results.npz") as data:
                    sigma = data['sigma']
                    eig_eval = data['eig_eval']
                    counteval = data['counteval']
        
            # Customize below
            np.savez(self.output_filename + "_CMA_ES_results",
                cost_ensemble_hist=self.cost_ensemble_hist,
                cost_ensemble_sigma_hist=self.cost_ensemble_sigma_hist,
                best_cost_hist=self.best_cost_hist,
                sigma_hist=self.sigma_hist,
                n_iter=self.n_iter,
                sigma=sigma,
                eig_eval=eig_eval,
                counteval=counteval,
            )
                        
            np.savez(self.output_filename + "_CMA_ES_density_hist",
                x_mean_hist=self.x_mean_hist,
                x_latent_hist=self.x_latent_hist,
                best_x_hist=self.best_x_hist,
                p_c=p_c,
                p_sigma=p_sigma,
                B=B,
                D=D,
                xmean=xmean,
                Cov=Cov,
            )