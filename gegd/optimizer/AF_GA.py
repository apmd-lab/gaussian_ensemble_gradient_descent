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
                 Nbatch,
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
        self.Nbatch = Nbatch
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
    
    def get_loss_batch(self, x):
        # Get Brush Binarized Densities ------------------------------------------------------------
        x_bin = dtf.binarize(x, self.symmetry, self.periodic, self.Nx, self.Ny, self.min_feature_size, self.brush_shape, self.beta_proj, self.sigma_filter,
                             upsample_ratio=self.upsample_ratio, padding=self.padding, method=self.feasible_design_generation_method, Nthreads=self.Nthreads)

        # Sample Modified Cost Function --------------------------------------------------------------
        self.cost_obj.set_accuracy(self.high_fidelity_setting)
        
        f_batch = np.zeros(self.Nbatch)
        for n in range(self.Nbatch):
            f_batch[n] = self.cost_obj.get_cost(x_bin[n,:], get_grad=False)
        
        return f_batch, x_bin

    def GA(
        self,
        lb,
        ub,
        survival_rate=0.5, # percentage of chromosomes that survives
        elitism_rate=0.1, # percentage of survivors that are preserved w/out mutation
        tournament_size=2, # number of genes in tournament selection
        mutation_rate=0.01, # mutation probability of each gene
        x=None,
    ):

        #Chromosome Initialization
        if x is None:
            x = np.random.uniform(lb, ub, size=(self.Nbatch, self.Ndim))

        while True:
            t1 = time.time()

            #Cost Evaluation
            cost, x_bin = self.get_loss_batch(x)
            
            #Natural Selection (sorting)
            sorted_index = np.argsort(cost)
            cost = cost[sorted_index]
            x = x[sorted_index,:]
            x_bin = x_bin[sorted_index,:]

            if self.best_cost_hist.size == 0:
                new_best = True
                self.best_cost_hist = np.append(self.best_cost_hist, cost[0])
            else:
                new_best = self.best_cost_hist[-1] > cost[0]
                self.best_cost_hist = np.append(self.best_cost_hist, np.min((self.best_cost_hist[-1], cost[0])))
                
            if self.best_chromosome_hist is None:
                self.best_chromosome_hist = x[0,:].copy()
            else:
                if new_best:
                    self.best_chromosome_hist = np.vstack((self.best_chromosome_hist, x[0,:]))
                else:
                    if self.best_chromosome_hist.ndim == 1:
                        self.best_chromosome_hist = np.vstack((self.best_chromosome_hist, self.best_chromosome_hist))
                    else:
                        self.best_chromosome_hist = np.vstack((self.best_chromosome_hist, self.best_chromosome_hist[-1,:]))

            if self.best_x_binary_hist is None:
                self.best_x_binary_hist = x_bin[0,:].copy()
            else:
                if new_best:
                    self.best_x_binary_hist = np.vstack((self.best_x_binary_hist, x_bin[0,:]))
                else:
                    if self.best_x_binary_hist.ndim == 1:
                        self.best_x_binary_hist = np.vstack((self.best_x_binary_hist, self.best_x_binary_hist))
                    else:
                        self.best_x_binary_hist = np.vstack((self.best_x_binary_hist, self.best_x_binary_hist[-1,:]))
            
            t2 = time.time()
            t_rem = (t2 - t1)*(self.maxiter - self.n_iter + 1)/3600

            if comm.rank == 0:
                print('    | %12d | %12.5f | %12.5f | %12.5f |   %5.2f   |' %(self.n_iter, np.mean(cost), np.std(cost), self.best_cost_hist[-1], t_rem), flush=True)

            self.save_data(x=x)

            self.n_iter += 1
                
            #Termination Condition
            if self.n_iter > self.maxiter:
                break
                
            for i in range(int(self.Nbatch * survival_rate * elitism_rate), self.Nbatch): # elites are passed to the next generation as is
                if i >= int(self.Nbatch * survival_rate): # chromosomes that failed to survive are replaced by offsprings of chromosomes that survived
                    #Tournament Selection (to select 2 parents for reproduction)
                    while True:
                        gene_ts_final = np.zeros(2).astype(int)
                        for j in range(2):
                            gene_ts = np.random.randint(0, int(self.Nbatch * survival_rate), size=(tournament_size))
                            cost_ts = cost[gene_ts]
                            sorted_index = np.argsort(cost_ts)
                            gene_ts_final[j] = gene_ts[sorted_index[0]]
                        
                        if gene_ts_final[0] != gene_ts_final[1]:
                            break
                            
                    #Reproduction
                    cut1 = np.random.randint(0, self.Ndim)
                    cut2 = np.random.randint(cut1 + 1, self.Ndim + 1)
                    x[i,:cut1] = x[gene_ts_final[0],:cut1]
                    x[i,cut1:cut2] = x[gene_ts_final[1],cut1:cut2]
                    x[i,cut2:] = x[gene_ts_final[0],cut2:]
                        
                #Mutation (chromosomes that survived but are not elites undergo mutations)
                for j in range(self.Ndim):
                    if np.random.random() < mutation_rate:
                        x[i,j] = np.random.uniform(lb, ub)

    def run(
        self,
        n_seed,
        output_filename,
        survival_rate=0.5,
        elitism_rate=0.1,
        tournament_size=2,
        mutation_rate=0.01,
        load_data=False,
    ):

        if comm.rank == 0:
            print('### Genetic Algorithm (seed = ' + str(n_seed) + ')\n', flush=True)
    
        np.random.seed(n_seed)
        self.output_filename = output_filename

        lb = -1.0
        ub = 1.0

        if load_data:
            data_file1 = output_filename + "_AF_GA_results.npz"
            data_file2 = output_filename + "_AF_GA_density_hist.npz"
                
            with np.load(data_file1) as data:
                self.n_iter = data['n_iter']
                self.best_cost_hist = data['best_cost_hist'][:self.n_iter]
                
            with np.load(data_file2) as data:
                self.best_chromosome_hist = data['best_chromosome_hist'][:self.n_iter,:]
                self.best_x_binary_hist = data['best_x_binary_hist'][:self.n_iter,:]
                
                x = data['x']

        else:
            self.best_cost_hist = np.zeros(0)
            self.best_chromosome_hist = None
            self.best_x_binary_hist = None
            
            x = None
            
            self.n_iter = 0
    
        if comm.rank == 0:
            print('    |  Iteration   |  Avg. Cost   | Cost Stdev.  |  Best Cost   | t_rem(hr) |', flush=True)
    
        self.GA(
            lb,
            ub,
            survival_rate=survival_rate,
            elitism_rate=elitism_rate,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            x=x,
        )
        
        self.save_data()
    
    def save_data(self, x=0):
        if comm.rank == 0:
            np.savez(
                self.output_filename + "_AF_GA_results",
                best_cost_hist=self.best_cost_hist,
                n_iter=self.n_iter,
            )
                     
            np.savez(
                self.output_filename + "_AF_GA_density_hist",
                best_chromosome_hist=self.best_chromosome_hist,
                best_x_binary_hist=self.best_x_binary_hist,
                x=x,
            )