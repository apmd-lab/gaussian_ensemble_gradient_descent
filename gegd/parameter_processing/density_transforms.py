import numpy as np
import os
from scipy.ndimage import gaussian_filter
import torch
import gegd.parameter_processing.symmetry_operations as symOp
import gegd.parameter_processing.feasible_design_generator.fdg as FDG
import gegd.parameter_processing.bilevel_design_generator as bdg
import time

from mpi4py import MPI
comm = MPI.COMM_WORLD

def filter_and_project(x, symmetry, periodic, Nx, Ny, min_feature_size, sigma_filter, beta_proj, padding=None):
    x_sym = symOp.symmetrize(x, symmetry, Nx, Ny)

    if sigma_filter is not None:
        if periodic:
            x_filter = gaussian_filter(x_sym, sigma=sigma_filter, mode='wrap')
        else:
            padding[min_feature_size:-min_feature_size,min_feature_size:-min_feature_size] = x_sym.copy()
            x_filter = gaussian_filter(padding, sigma=sigma_filter, mode='constant', cval=0)[min_feature_size:-min_feature_size,min_feature_size:-min_feature_size]
    else:
        x_filter = x_sym.copy()
    
    x_desym = symOp.desymmetrize(x_filter, symmetry, Nx, Ny)
    
    if beta_proj == np.inf:
        x_proj = x_desym.copy()
        x_proj[x_desym<=0] = -1
        x_proj[x_desym>0] = 1
    else:
        x_proj = np.tanh(beta_proj*x_desym)
    
    return x_proj
    
def backprop_filter_and_project(jac_sym, x_latent, symmetry, periodic, Nx, Ny, min_feature_size, sigma_filter, beta_proj, padding=None):
    x_sym = symOp.symmetrize(x_latent, symmetry, Nx, Ny)

    if sigma_filter is not None:
        if periodic:
            x_filter = gaussian_filter(x_sym, sigma=sigma_filter, mode='wrap')
        else:
            padding[min_feature_size:-min_feature_size,min_feature_size:-min_feature_size] = x_sym.copy()
            x_filter = gaussian_filter(padding, sigma=sigma_filter, mode='constant', cval=0)[min_feature_size:-min_feature_size,min_feature_size:-min_feature_size]
    else:
        x_filter = x_sym.copy()

    jac_proj = beta_proj/np.cosh(beta_proj*x_filter)**2
    jac_proj *= jac_sym
    
    if sigma_filter is not None:
        if periodic:
            jac_filter = gaussian_filter(jac_proj, sigma=sigma_filter, mode='wrap')
        else:
            jac_filter = gaussian_filter(jac_proj, sigma=sigma_filter, mode='constant', cval=0)
        
    jac_desym = symOp.desymmetrize_jacobian(jac_filter, symmetry, Nx, Ny)
    
    return jac_desym

def binarize(
    x,
    symmetry,
    periodic,
    Nx,
    Ny,
    min_feature_size,
    brush_shape,
    beta_proj,
    sigma_filter,
    dx=None,
    upsample_ratio=1,
    padding=None,
    method='brush',
    output_details=False,
    Nthreads=1,
    print_runtime_details=False,
):
    if method == 'brush':
        if padding is None:
            x_reward = np.zeros((N_designs, Nx, Ny)).astype(np.float32)
        else:
            x_reward = np.zeros((N_designs, Nx + 2*min_feature_size, Ny + 2*min_feature_size)).astype(np.float32)

        for n in range(N_designs):
            if dx is not None:
                x_fp = filter_and_project(x, symmetry, periodic, Nx, Ny, min_feature_size, sigma_filter, beta_proj, padding=padding).reshape(-1) + dx[n,:]
            else:
                x_fp = filter_and_project(x[n,:], symmetry, periodic, Nx, Ny, min_feature_size, sigma_filter, beta_proj, padding=padding)
        
            if padding is None:
                x_reward[n,:,:] = symOp.symmetrize(x_fp, symmetry, Nx, Ny)
            else:
                x_reward[n,:,:] = padding.copy()
                x_reward[n,min_feature_size:-min_feature_size,min_feature_size:-min_feature_size] = symOp.symmetrize(x_fp, symmetry, Nx, Ny)
        
        t1 = time.time()
        try:
            if min_feature_size is not None:
                if Nthreads > 0:
                    if upsample_ratio == 1:
                        x_brush = FDG.make_feasible_parallel(
                            x_reward,
                            min_feature_size,
                            periodic,
                            symmetry,
                            2,
                            upsample_ratio,
                            Nthreads).reshape(N_designs, -1)
                        
                    else:
                        x_brush_lowres = FDG.make_feasible_parallel(
                            x_reward,
                            min_feature_size,
                            periodic,
                            symmetry,
                            2,
                            1,
                            Nthreads).reshape(N_designs, Nx, Ny)
                        
                        x_brush = FDG.make_feasible_parallel(
                            2*x_brush_lowres.astype(np.float32)-1,
                            min_feature_size,
                            periodic,
                            symmetry,
                            2,
                            upsample_ratio,
                            Nthreads).reshape(N_designs, -1)
                
                else:
                    quo, rem = divmod(N_designs, comm.size)
                    data_size = np.array([quo + 1 if p < rem else quo for p in range(comm.size)]).astype(np.int32)
                    data_disp = np.array([sum(data_size[:p]) for p in range(comm.size + 1)])
                    x_reward_proc = x_reward[data_disp[comm.rank]:data_disp[comm.rank+1],:,:]
                    
                    if data_size[comm.rank] > 0:
                        if upsample_ratio == 1:
                            x_brush_proc = FDG.make_feasible_parallel(
                                x_reward_proc,
                                min_feature_size,
                                periodic,
                                symmetry,
                                2,
                                upsample_ratio,
                                1).reshape(data_size[comm.rank], -1).astype(np.float64)
                            
                        else:
                            x_brush_lowres_proc = FDG.make_feasible_parallel(
                                x_reward_proc,
                                min_feature_size,
                                periodic,
                                symmetry,
                                2,
                                1,
                                1).reshape(data_size[comm.rank], Nx, Ny)
                            
                            x_brush_proc = FDG.make_feasible_parallel(
                                2*x_brush_lowres_proc.astype(np.float32)-1,
                                min_feature_size,
                                periodic,
                                symmetry,
                                2,
                                upsample_ratio,
                                1).reshape(data_size[comm.rank], -1).astype(np.float64)
                    else:
                        x_brush_proc = np.array([]).astype(np.float64)
                    
                    if padding is None:
                        data_size_temp = data_size*Nx*Ny*upsample_ratio**2
                        data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)])
                        
                        x_brush_temp = np.zeros(N_designs*Nx*Ny*upsample_ratio**2)
                        comm.Allgatherv(x_brush_proc.reshape(-1), [x_brush_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
                        x_brush = x_brush_temp.reshape(N_designs, -1)
                    else:
                        data_size_temp = data_size*(Nx + 2*min_feature_size)*(Ny + 2*min_feature_size)*upsample_ratio**2
                        data_disp_temp = np.array([sum(data_size_temp[:p]) for p in range(comm.size)])
                        
                        x_brush_temp = np.zeros(N_designs*(Nx + 2*min_feature_size)*(Ny + 2*min_feature_size)*upsample_ratio**2)
                        comm.Allgatherv(x_brush_proc.reshape(-1), [x_brush_temp, data_size_temp, data_disp_temp, MPI.DOUBLE])
                        x_brush = x_brush_temp.reshape(N_designs, -1)
        except:
            if comm.rank == 0:
                np.savez(os.path.join(os.path.expanduser("~"), "debug_brush"), x_sym=x_sym)
            assert False
        
        t2 = time.time()

        if print_runtime_details:
            if comm.rank == 0:
                print("--> Brush Generator Runtime: " + str(t2 - t1) + " s", flush=True)
        
        if padding is not None:
            x_brush_crop = x_brush.reshape(N_designs, (Nx + 2*min_feature_size)*upsample_ratio, (Ny + 2*min_feature_size)*upsample_ratio)
            x_brush_crop = x_brush_crop[:,min_feature_size:-min_feature_size,min_feature_size:-min_feature_size]
            x_brush = x_brush_crop.reshape(N_designs, -1)

    elif method == 'two_phase_projection':
        if dx is not None:
            N_designs = dx.shape[0]
            x_reward = np.zeros_like(dx).astype(np.float64)
        else:
            N_designs = x.shape[0]
            x_reward = np.zeros_like(x).astype(np.float64)
        
        for n in range(N_designs):
            if dx is not None:
                x_reward[n,:] = filter_and_project(x, symmetry, periodic, Nx, Ny, min_feature_size, sigma_filter, beta_proj, padding=padding).reshape(-1) + dx[n,:]
            else:
                x_reward[n,:] = filter_and_project(x[n,:], symmetry, periodic, Nx, Ny, min_feature_size, sigma_filter, beta_proj, padding=padding)

        t1 = time.time()
        generator = bdg.conditional_generator(
            Nx,
            Ny,
            symmetry,
            periodic,
            padding,
            min_feature_size,
            maxiter=60,
        )

        x_reward = torch.tensor(x_reward, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float64)
        x_near_binary = generator.generate_near_binary_no_grad(x_reward)
        x_bin = np.where(x_near_binary.detach().cpu().numpy() < 0.5, 0, 1).reshape(N_designs, -1).astype(np.float32)

        t2 = time.time()

        if print_runtime_details:
            if comm.rank == 0:
                print("--> Two Phase Projection Generator Runtime: " + str(t2 - t1) + " s", flush=True)
    
    if output_details:
        return np.squeeze(x_reward), np.squeeze(x_bin)
    
    else:
        return np.squeeze(x_bin)