import numpy as np
import torch
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
                 cuda_ind=0,
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
        self.cuda_ind = cuda_ind
        
        # Get Number of Independent Parameters
        if symmetry == 0:
            self.Ndim = Nx*Ny
           
        elif symmetry == 1:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*Ny)
        
        elif symmetry == 2:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*np.floor(Ny/2 + 0.5))
        
        elif symmetry == 4:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*(np.floor(Nx/2 + 0.5) + 1)/2)
        
        self.device = torch.device(f'cuda:{cuda_ind}') if torch.cuda.is_available() else torch.device('cpu')
    
    def get_cost_samples(self, x):
        # Get Brush Binarized Densities ------------------------------------------------------------
        x_bin = dtf.binarize(
            x,
            self.symmetry,
            self.periodic,
            self.Nx,
            self.Ny,
            self.min_feature_size,
            self.brush_shape,
            self.beta_proj,
            self.sigma_filter,
            upsample_ratio=self.upsample_ratio,
            padding=self.padding,
            method=self.feasible_design_generation_method,
            Nthreads=self.Nthreads,
            cuda_ind=self.cuda_ind,
            )

        # Sample Modified Cost Function --------------------------------------------------------------
        self.cost_obj.set_accuracy(self.high_fidelity_setting)
        
        f_batch = np.zeros(self.Nsample)
        for n in range(self.Nsample):
            f_batch[n] = self.cost_obj.get_cost(x_bin[n,:], get_grad=False)
        
        return f_batch, x_bin

    def sep_CMA_ES(
        self,
        xmean=None,
        sigma=None,
        Cov=None,
        p_c=None,
        p_sigma=None,
        D=None,
    ):
        # Set Selection Parameters
        #self.Nsample = 25 #int(4 + np.floor(3 * np.log(self.Ndim)))
        mu = torch.tensor(int(np.floor(self.Nsample / 2)), dtype=torch.int64, device=self.device)
        mu_arange = torch.arange(mu, dtype=torch.float64, device=self.device) + 1
        weights = (torch.log(mu + 1) - torch.log(mu_arange)) / torch.sum(torch.log(mu + 1) - torch.log(mu_arange))
        weights /= torch.sum(weights)
        mu_eff = 1 / torch.sum(weights**2)

        # Set Adaptation Parameters
        c_c = 4 / (self.Ndim + 4)
        c_sigma = (mu_eff + 2) / (self.Ndim + mu_eff + 3)
        c_cov = ((self.Ndim + 2) / 3) * (1 / mu_eff) * (2 / (self.Ndim + torch.sqrt(torch.tensor(2, dtype=torch.float64, device=self.device)))**2) \
            + (1 - 1 / mu_eff) * torch.min(torch.tensor([1, (2 * mu_eff - 1) / ((self.Ndim + 2)**2 + mu_eff)]))
        d_sigma = 1 + 2 * torch.max(torch.tensor([0, torch.sqrt((mu_eff - 1) / (self.Ndim + 1)) - 1])) + c_sigma

        # Initialize Dynamic Internal Parameters
        p_c = torch.tensor(p_c, dtype=torch.float64, device=self.device) if p_c is not None else torch.zeros(self.Ndim, dtype=torch.float64, device=self.device)
        p_sigma = torch.tensor(p_sigma, dtype=torch.float64, device=self.device) if p_sigma is not None else torch.zeros(self.Ndim, dtype=torch.float64, device=self.device)
        B = torch.eye(self.Ndim, dtype=torch.float64, device=self.device)
        D = torch.diag(D, dtype=torch.float64, device=self.device) if D is not None else torch.eye(self.Ndim, dtype=torch.float64, device=self.device)
        xmean = torch.tensor(xmean, dtype=torch.float64, device=self.device) if xmean is not None else torch.zeros(self.Ndim, dtype=torch.float64, device=self.device)
        sigma = torch.tensor(sigma, dtype=torch.float64, device=self.device) if sigma is not None else torch.tensor(0.1, dtype=torch.float64, device=self.device)
        Cov = torch.diag(Cov, dtype=torch.float64, device=self.device) if Cov is not None else torch.eye(self.Ndim, dtype=torch.float64, device=self.device)
        chi_N = self.Ndim**0.5 * (1 - 1 / (4 * self.Ndim) + 1 / (21 * self.Ndim**2))

        while True:
            t1 = time.time()

            # Sample Candidates
            arz = np.random.normal(size=(self.Ndim, self.Nsample))
            arz = torch.tensor(arz, dtype=torch.float64, device=self.device)
            arx = xmean.unsqueeze(-1) + sigma * (B @ D @ arz)
            
            # Cost Evaluation
            cost, x_bin = self.get_cost_samples(arx.T.detach().cpu().numpy())
            sorted_index = np.argsort(cost)
            sorted_index_tensor = torch.tensor(sorted_index, dtype=torch.int64, device=self.device)
            cost = cost[sorted_index]
            arx = arx[:,sorted_index_tensor]
            x_bin = x_bin[sorted_index,:]

            if self.x_mean_hist is None:
                self.x_mean_hist = xmean.detach().cpu().numpy().copy()
            else:
                self.x_mean_hist = np.vstack((self.x_mean_hist, xmean.detach().cpu().numpy()))

            if self.x_latent_hist is None:
                self.x_latent_hist = arx[:,0].detach().cpu().numpy().copy()
            else:
                self.x_latent_hist = np.vstack((self.x_latent_hist, arx[:,0].detach().cpu().numpy()))
            
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
            self.sigma_hist = np.append(self.sigma_hist, sigma.item())
            
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
                p_c=p_c.detach().cpu().numpy(),
                p_sigma=p_sigma.detach().cpu().numpy(),
                D=torch.diag(D).detach().cpu().numpy(),
                xmean=xmean.detach().cpu().numpy(),
                sigma=sigma.item(),
                Cov=torch.diag(Cov).detach().cpu().numpy(),
            )

            if self.n_iter > self.maxiter:
                self.n_iter += 1
                break

            # Update Mean
            xmean = arx[:,:mu] @ weights
            zmean = arz[:,:mu] @ weights

            # Update Evolution Path and Step Size
            p_sigma = (1 - c_sigma) * p_sigma + torch.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ zmean)
            h_sigma = torch.linalg.norm(p_sigma) / torch.sqrt(1 - (1 - c_sigma)**(2 * (self.n_iter + 1))) < (1.4 + 2 / (self.Ndim + 1)) * chi_N
            h_sigma = h_sigma.to(dtype=torch.float64, device=self.device)
            p_c = (1 - c_c) * p_c + h_sigma * torch.sqrt(c_c * (2 - c_c) * mu_eff) * (B @ D @ zmean)

            # Update Covariance Matrix
            Cov = (1 - c_cov) * Cov \
                + (c_cov / mu_eff) * p_c.unsqueeze(-1) @ p_c.unsqueeze(0)
            BDz = torch.diag(D).unsqueeze(-1) * arz[:,:mu]
            for i in range(mu):
                Cov += c_cov * (1 - 1 / mu_eff) * weights[i] * BDz[:,i].unsqueeze(-1) @ BDz[:,i].unsqueeze(0)
            
            # Update Step Size
            sigma = sigma * torch.exp((c_sigma / d_sigma) * (torch.linalg.norm(p_sigma) / chi_N - 1))

            # Update B and D
            D = torch.diag(torch.sqrt(torch.diag(Cov)))
            
            self.n_iter += 1

    def run(self, n_seed, output_filename, x0=None, load_data=False):
        if comm.rank == 0:
            print('### sep-CMA-ES (seed = ' + str(n_seed) + ')\n', flush=True)
    
        np.random.seed(n_seed)
        self.output_filename = output_filename
        
        if load_data:
            data_file1 = output_filename + "_sep_CMA_ES_results.npz"
            data_file2 = output_filename + "_sep_CMA_ES_density_hist.npz"
                
            with np.load(data_file1) as data:
                self.n_iter = data['n_iter']
                
                self.cost_ensemble_hist = data['cost_ensemble_hist'][:self.n_iter]
                self.best_cost_hist = data['best_cost_hist'][:self.n_iter]
                self.cost_ensemble_sigma_hist = data['cost_ensemble_sigma_hist'][:self.n_iter]
                self.sigma_hist = data['sigma_hist'][:self.n_iter]

                sigma = data['sigma']
                
            with np.load(data_file2) as data:
                self.x_latent_hist = data['x_latent_hist'][:self.n_iter,:]
                self.x_mean_hist = data['x_mean_hist'][:self.n_iter,:]
                self.best_x_hist = data['best_x_hist'][:self.n_iter,:]
                
                p_c = data['p_c']
                p_sigma = data['p_sigma']
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
            D = None
        
        if comm.rank == 0:
            print('    | Iter |   N   |  sigma  |  Cost Ens  | Cost Best | t_rem(hr) |',
                  flush=True)

        self.sep_CMA_ES(
            xmean=x0,
            sigma=sigma,
            Cov=Cov,
            p_c=p_c,
            p_sigma=p_sigma,
            D=D,
        )
        
        self.save_data()
    
    def save_data(
        self,
        p_c=None,
        p_sigma=None,
        D=None,
        xmean=None,
        sigma=None,
        Cov=None,
    ):

        if comm.rank == 0:
            if p_c is None:
                with np.load(self.output_filename + "_sep_CMA_ES_density_hist.npz") as data:
                    p_c = data['p_c']
                    p_sigma = data['p_sigma']
                    D = data['D']
                    xmean = data['xmean']
                    Cov = data['Cov']

            if sigma is None:
                with np.load(self.output_filename + "_sep_CMA_ES_results.npz") as data:
                    sigma = data['sigma']
        
            if self.x_mean_hist.ndim == 1:
                x_mean_final = self.x_mean_hist.copy()
                x_latent_final = self.x_latent_hist.copy()
                best_x_final = self.best_x_hist.copy()
            else:
                x_mean_final = self.x_mean_hist[-1,:]
                x_latent_final = self.x_latent_hist[-1,:]
                best_x_final = self.best_x_hist[-1,:]

            # Customize below
            np.savez(self.output_filename + "_sep_CMA_ES_results",
                cost_ensemble_hist=self.cost_ensemble_hist,
                cost_ensemble_sigma_hist=self.cost_ensemble_sigma_hist,
                best_cost_hist=self.best_cost_hist,
                sigma_hist=self.sigma_hist,
                x_mean_final=x_mean_final,
                x_latent_final=x_latent_final,
                best_x_final=best_x_final,
                n_iter=self.n_iter,
                sigma=sigma,
            )
                        
            np.savez(self.output_filename + "_sep_CMA_ES_density_hist",
                x_mean_hist=self.x_mean_hist,
                x_latent_hist=self.x_latent_hist,
                best_x_hist=self.best_x_hist,
                p_c=p_c,
                p_sigma=p_sigma,
                D=D,
                xmean=xmean,
                Cov=Cov,
            )