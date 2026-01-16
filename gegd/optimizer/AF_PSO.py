import numpy as np
import gegd.parameter_processing.density_transforms as dtf
import time

from mpi4py import MPI
comm = MPI.COMM_WORLD

class optimizer:
    def __init__(self,
                 Nx,
                 Ny,
                 Nswarm,
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
        self.Nswarm = Nswarm
        self.symmetry = symmetry
        self.periodic = periodic
        self.padding = padding
        self.maxiter = maxiter
        self.beta_proj = beta_proj
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

    def get_loss_swarm(self, x):
        # Get Brush Binarized Densities ------------------------------------------------------------
        x_bin = dtf.binarize(x, self.symmetry, self.periodic, self.Nx, self.Ny, self.min_feature_size, self.brush_shape, self.beta_proj, self.sigma_filter,
                             upsample_ratio=self.upsample_ratio, padding=self.padding, method=self.feasible_design_generation_method, Nthreads=self.Nthreads)

        # Sample Modified Cost Function --------------------------------------------------------------
        self.cost_obj.set_accuracy(self.high_fidelity_setting)
        
        f_swarm = np.zeros(self.Nswarm)
        for n in range(self.Nswarm):
            f_swarm[n] = self.cost_obj.get_cost(x_bin[n,:], get_grad=False)
        
        return f_swarm, x_bin

    def PSO(self,
            lb,
            ub,
            coeff_cognitive=1.49,
            coeff_social=1.49,
            coeff_inertia=0.9,
            velocity=None,
            x=None,
            pbest_loss=None,
            pbest_x=None,
            pbest_x_bin=None,
            ):
    
        vmax = ub - lb
        if velocity is None and x is None:
            #Latin Hypercube Particle Initialization
            increment = (ub - lb)/self.Nswarm

            x = np.zeros((self.Nswarm, self.Ndim))
            velocity = np.zeros((self.Nswarm, self.Ndim))
            for nd in range(self.Ndim):
                swarm_ind = np.random.permutation(self.Nswarm).reshape(self.Nswarm)
                if lb[nd] == ub[nd]:
                    x[:,nd] = lb[nd]*np.ones(self.Nswarm)
                    velocity[:,nd] = np.zeros(self.Nswarm)
                else:
                    x[:,nd] = lb[nd] + (swarm_ind + 1)*increment[nd] - increment[nd]*np.random.rand(self.Nswarm)
                    velocity[:,nd] = -vmax[nd] + 2*vmax[nd]*np.random.rand(self.Nswarm)
    
        if pbest_loss is None and pbest_x is None:
            # Initialize pbest
            pbest_loss = np.zeros(self.Nswarm)
            pbest_x = np.zeros((self.Nswarm, self.Ndim))
    
        while True:
            t1 = time.time()
            if self.n_iter > self.maxiter:
                break
        
            loss_swarm, x_bin = self.get_loss_swarm(x)
            
            #Particle Best
            if self.n_iter == 0:
                pbest_loss = loss_swarm.copy()
                pbest_x = x.copy()
                pbest_x_bin = x_bin.copy()
            else:
                pbest_mask = loss_swarm < pbest_loss
                pbest_loss[pbest_mask] = loss_swarm[pbest_mask]
                pbest_x[pbest_mask,:] = x[pbest_mask,:]
                pbest_x_bin[pbest_mask,:] = x_bin[pbest_mask,:]
            
            #Global Best
            gbest_ind = np.argmin(pbest_loss)
            if self.gbest_x_hist is None:
                self.gbest_loss_hist = np.append(self.gbest_loss_hist, pbest_loss[gbest_ind])
                self.gbest_x_hist = pbest_x[gbest_ind,:].copy()
                self.gbest_x_binary_hist = pbest_x_bin[gbest_ind,:].copy()
            else:
                if self.gbest_loss_hist[-1] > pbest_loss[gbest_ind]:
                    self.gbest_loss_hist = np.append(self.gbest_loss_hist, pbest_loss[gbest_ind])
                    self.gbest_x_hist = np.vstack((self.gbest_x_hist, pbest_x[gbest_ind,:]))
                    self.gbest_x_binary_hist = np.vstack((self.gbest_x_binary_hist, pbest_x_bin[gbest_ind,:]))
                else:
                    self.gbest_loss_hist = np.append(self.gbest_loss_hist, self.gbest_loss_hist[-1])
                    if self.gbest_x_hist.ndim == 1:
                        self.gbest_x_hist = np.vstack((self.gbest_x_hist, self.gbest_x_hist))
                        self.gbest_x_binary_hist = np.vstack((self.gbest_x_binary_hist, self.gbest_x_binary_hist))
                    else:
                        self.gbest_x_hist = np.vstack((self.gbest_x_hist, self.gbest_x_hist[-1,:]))
                        self.gbest_x_binary_hist = np.vstack((self.gbest_x_binary_hist, self.gbest_x_binary_hist[-1,:]))

            t2 = time.time()
            t_rem = (t2 - t1)*(self.maxiter - self.n_iter + 1)/3600

            if comm.rank == 0:
                print('    | %12d | %12.5f | %12.5f | %12.5f |   %5.2f   |' %(self.n_iter, np.mean(pbest_loss), np.std(pbest_loss), self.gbest_loss_hist[-1], t_rem), flush=True)
            self.save_data(coeff_inertia=coeff_inertia,
                           velocity=velocity,
                           x=x,
                           pbest_loss=pbest_loss,
                           pbest_x=pbest_x,
                           pbest_x_bin=pbest_x_bin)

            #Parameter Updates
            if self.gbest_loss_hist.size >= 5 and self.gbest_loss_hist[-1] == self.gbest_loss_hist[-5]:
                coeff_inertia *= 0.95
            
            #Velocity Updates
            if self.gbest_x_hist.ndim == 1:
                velocity = coeff_inertia*velocity + coeff_cognitive*np.random.rand()*(pbest_x - x) + coeff_social*np.random.rand()*(self.gbest_x_hist[np.newaxis,:] - x)
            else:
                velocity = coeff_inertia*velocity + coeff_cognitive*np.random.rand()*(pbest_x - x) + coeff_social*np.random.rand()*(self.gbest_x_hist[-1,:][np.newaxis,:] - x)
            
            vmax_violation = np.abs(velocity) - vmax[np.newaxis,:]
            vmax_violation_mask = np.sum(np.abs(velocity) > vmax[np.newaxis,:], axis=1).astype(bool)
            vmax_violation_dim = np.argmax(vmax_violation, axis=1)[vmax_violation_mask]
            velocity[vmax_violation_mask,:] *= (vmax[vmax_violation_dim]/np.max(np.abs(velocity[vmax_violation_mask,:]), axis=1))[:,np.newaxis]
            
            #Craziness
            N_cr = np.random.permutation(self.Nswarm)[:int(self.Nswarm/10)]
            if np.random.rand() < 0.22:
                for t in range(int(self.Nswarm/10)):
                    velocity[N_cr[t],:] = -vmax + vmax*np.random.rand(self.Ndim)
                    
            #Position Updates
            x += velocity
            
            lb_violation_mask = x < lb[np.newaxis,:]
            ub_violation_mask = x > ub[np.newaxis,:]
            x_lb = lb[np.newaxis,:] + (lb[np.newaxis,:] - x)
            x_ub = ub[np.newaxis,:] - (x - ub[np.newaxis,:])
            x[lb_violation_mask] = x_lb[lb_violation_mask]
            x[ub_violation_mask] = x_ub[ub_violation_mask]
            
            self.n_iter += 1

    def run(self, n_seed, output_filename, coeff_cognitive=1.49, coeff_social=1.49, coeff_inertia=0.9, load_data=False):
        if comm.rank == 0:
            print('### Particle Swarm Optimization (seed = ' + str(n_seed) + ')\n', flush=True)
    
        np.random.seed(n_seed)
        self.output_filename = output_filename

        lb = -np.ones(self.Ndim)
        ub = np.ones(self.Ndim)

        if load_data:
            data_file1 = output_filename + "_AF_PSO_results.npz"
            data_file2 = output_filename + "_AF_PSO_density_hist.npz"
                
            with np.load(data_file1) as data:
                self.n_iter = data['n_iter']
                self.gbest_loss_hist = data['gbest_loss_hist'][:self.n_iter]

                coeff_inertia = data['coeff_inertia']
                pbest_loss = data['pbest_loss']
                
            with np.load(data_file2) as data:
                self.gbest_x_hist = data['gbest_x_hist'][:self.n_iter,:]
                self.gbest_x_binary_hist = data['gbest_x_binary_hist'][:self.n_iter,:]
                
                velocity = data['velocity']
                x = data['x']
                pbest_x = data['pbest_x']
                pbest_x_bin = data['pbest_x_bin']

        else:
            self.gbest_loss_hist = np.zeros(0)
            self.gbest_x_hist = None
            self.gbest_x_binary_hist = None
            
            velocity = None
            x = None
            pbest_loss = None
            pbest_x = None
            pbest_x_bin = None
            
            self.n_iter = 0
    
        if comm.rank == 0:
            print('    |  Iteration   |  PBest Avg.  | PBest Stdev. |  GBest Cost  | t_rem(hr) |', flush=True)
    
        self.PSO(lb, ub, coeff_inertia=coeff_inertia,
                 velocity=velocity, x=x, pbest_loss=pbest_loss, pbest_x=pbest_x, pbest_x_bin=pbest_x_bin)
        
        self.save_data()
    
    def save_data(self, coeff_inertia=0, velocity=0, x=0, pbest_loss=0, pbest_x=0, pbest_x_bin=0):
        if comm.rank == 0:
            np.savez(self.output_filename + "_AF_PSO_results",
                     gbest_loss_hist=self.gbest_loss_hist,
                     n_iter=self.n_iter,
                     coeff_inertia=coeff_inertia,
                     pbest_loss=pbest_loss)
                     
            np.savez(self.output_filename + "_AF_PSO_density_hist",
                     gbest_x_hist=self.gbest_x_hist,
                     gbest_x_binary_hist=self.gbest_x_binary_hist,
                     velocity=velocity,
                     x=x,
                     pbest_x=pbest_x,
                     pbest_x_bin=pbest_x_bin)