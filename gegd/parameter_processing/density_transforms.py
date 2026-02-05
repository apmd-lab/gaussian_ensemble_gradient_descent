import numpy as np
import os
from scipy.ndimage import gaussian_filter
import torch
import gegd.parameter_processing.symmetry_operations as symOp
try:
    import gegd.parameter_processing.feasible_design_generator.fdg as FDG
except ImportError:
    import gegd.parameter_processing.feasible_design_generator_python as FDG
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
    elif beta_proj == 0:
        x_proj = x_desym.copy()
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

    if beta_proj == 0:
        jac_proj = jac_sym.copy()
    else:
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
    Nthreads=1,
    print_runtime_details=False,
    cuda_ind=0,
):
    if method == 'brush':
        if dx is not None:
            N_designs = dx.shape[0]
        else:
            N_designs = x.shape[0]

        if padding is None:
            x_reward = np.zeros((N_designs, Nx, Ny)).astype(np.float32)
        else:
            x_reward = np.zeros((N_designs, Nx + 2*min_feature_size, Ny + 2*min_feature_size)).astype(np.float32)

        for n in range(N_designs):
            if padding is None:
                x_reward[n,:,:] = symOp.symmetrize(x[n,:], symmetry, Nx, Ny)
            else:
                x_reward[n,:,:] = padding.copy()
                x_reward[n,min_feature_size:-min_feature_size,min_feature_size:-min_feature_size] = symOp.symmetrize(x[n,:], symmetry, Nx, Ny)
        
        t1 = time.time()
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
        else:
            N_designs = x.shape[0]

        if dx is not None:
            x_reward = x + dx
        else:
            x_reward = x.copy()

        #cnt = 0
        #while os.path.exists('x_reward' + str(cnt) + '.npz'):
        #    cnt += 1
        #np.savez('x_reward' + str(cnt) + '.npz', x_reward=x_reward)

        t1 = time.time()
        generator = bdg.conditional_generator(
            Nx,
            Ny,
            symmetry,
            periodic,
            padding,
            min_feature_size,
            maxiter=60,
            cuda_ind=cuda_ind,
        )

        x_reward = torch.tensor(x_reward, device='cuda:' + str(cuda_ind) if torch.cuda.is_available() else 'cpu', dtype=torch.float64)
        x_near_binary = generator.generate_near_binary_no_grad(x_reward)
        x_bin = np.where(x_near_binary.detach().cpu().numpy() < 0.5, 0, 1).reshape(N_designs, -1).astype(np.float32)

        t2 = time.time()

        if print_runtime_details:
            if comm.rank == 0:
                print("--> Two Phase Projection Generator Runtime: " + str(t2 - t1) + " s", flush=True)

    return np.squeeze(x_bin)