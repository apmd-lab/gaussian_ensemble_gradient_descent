import torch
import torch.nn.functional as F
import numpy as np
import time

def conic_filter(mfs):
    X, Y = torch.meshgrid(torch.arange(-(mfs - 1) / 2, (mfs + 1) / 2), torch.arange(-(mfs - 1) / 2, (mfs + 1) / 2), indexing='ij')
    h = mfs / 2 - torch.sqrt(X**2 + Y**2) / (mfs / 2)
    h /= torch.sum(h)
    h[(X**2 + Y**2) > (mfs / 2)**2] = 0

    return h

def symmetrize_tensor(x, Nx, Ny, symmetry, device):
    # Detect batch mode
    batch_processing = x.ndim == 2
    if not batch_processing:
        # Standard case: x is (Ndim,)
        # Reshape to (1, Ndim) for unified processing, then squeeze at end
        x = x.unsqueeze(0)
    
    # Now x is (Batch, Ndim)
    Nbatch = x.shape[0]

    if symmetry == 0:
        x_sym = x.reshape(Nbatch, Nx, Ny)
    
    elif symmetry == 1:
        x_half = x.reshape(Nbatch, int(np.floor(Nx/2 + 0.5)), Ny)
        
        if Nx % 2 == 0:
            x_sym = torch.cat([x_half, torch.flip(x_half, [1])], dim=1)
        else:
            x_sym = torch.cat([x_half, torch.flip(x_half[:,:-1,:], [1])], dim=1)
        
    elif symmetry == 2:
        x_quart = x.reshape(Nbatch, int(np.floor(Nx/2 + 0.5)), int(np.floor(Ny/2 + 0.5)))
        
        if Ny % 2 == 0:
            top = torch.cat([x_quart, torch.flip(x_quart, [2])], dim=2)
        else:
            top = torch.cat([x_quart, torch.flip(x_quart[:,:,:-1], [2])], dim=2)
            
        if Nx % 2 == 0:
            x_sym = torch.cat([top, torch.flip(top, [1])], dim=1)
        else:
            x_sym = torch.cat([top, torch.flip(top[:,:-1,:], [1])], dim=1)
        
    elif symmetry == 4:
        assert Nx == Ny

        # Flatten just the spatial part for logic below if needed, but here we process batch
        # x is (B, Ndim)
        
        if Nx % 2 == 0:
            dim_half = int(Nx/2)
            x_quart = torch.zeros((Nbatch, dim_half, dim_half), device=device, dtype=x.dtype)
            rows, cols = torch.triu_indices(dim_half, dim_half, device=device)

            x_quart[:,rows,cols] = x
            
            # Symmetrize the quarter matrix
            # M = M + M.T - diag
            # For batch: M.transpose(1,2)
            x_quart = x_quart + x_quart.transpose(1, 2) - torch.diag_embed(torch.diagonal(x_quart, dim1=-2, dim2=-1))
            
            top_left = x_quart
            bottom_left = torch.rot90(x_quart, 1, [1, 2])
            bottom_right = torch.rot90(x_quart, 2, [1, 2])
            top_right = torch.rot90(x_quart, 3, [1, 2])
            
            top = torch.cat([top_left, top_right], dim=2)
            bottom = torch.cat([bottom_left, bottom_right], dim=2)
            x_sym = torch.cat([top, bottom], dim=1)

        else:
            dim_half = int((Nx + 1) / 2)
            x_quart = torch.zeros((Nbatch, dim_half, dim_half), device=device, dtype=x.dtype)
            rows, cols = torch.triu_indices(dim_half, dim_half, device=device)
            
            x_quart[:,rows,cols] = x
            x_quart = x_quart + x_quart.transpose(1, 2) - torch.diag_embed(torch.diagonal(x_quart, dim1=-2, dim2=-1))
            
            top_left = x_quart
            top_right = torch.rot90(x_quart, 3, [1, 2])[:,:,1:]
            bottom_left = torch.rot90(x_quart, 1, [1, 2])[:,1:,:]
            bottom_right = torch.rot90(x_quart, 2, [1, 2])[:,1:,1:]
            
            top = torch.cat([top_left, top_right], dim=2)
            bottom = torch.cat([bottom_left, bottom_right], dim=2)
            x_sym = torch.cat([top, bottom], dim=1)

    if not batch_processing:
        x_sym = x_sym.squeeze(0)
        
    return x_sym

def two_phase_projection(
        x,
        Nx,
        Ny,
        symmetry,
        periodic,
        padding,
        mfs,
        device,
        beta_proj=8,
        alpha_s=0.002,
        alpha_v=0.002,
    ):
    
    # Detect batch mode
    # x is either (Nx, Ny) or (B, Nx, Ny)
    batch_processing = x.ndim == 3
    
    beta_proj = torch.tensor(beta_proj, device=device, dtype=torch.float64)
    alpha_s = torch.tensor(alpha_s, device=device, dtype=torch.float64)
    alpha_v = torch.tensor(alpha_v, device=device, dtype=torch.float64)

    n_s = -torch.log(alpha_s)
    n_v = -torch.log(alpha_v)
    
    # Broadcast scalar/tensor ops works automatically for extra batch dim
    w_s = (1 + alpha_s) / (1 + alpha_s * torch.exp(2 * n_s * (1 - x)))
    w_v = -(1 + alpha_v) / (1 + alpha_v * torch.exp(2 * n_v * (1 + x)))
    
    kernel = conic_filter(mfs).to(dtype=torch.float64)
    padding_amount = kernel.shape[0] // 2
    
    if periodic:
        if batch_processing:
            # (B, 1, Nx, Ny)
            w_s_input = w_s.unsqueeze(1)
            w_v_input = w_v.unsqueeze(1)
        else:
            # (1, 1, Nx, Ny)
            w_s_input = w_s.unsqueeze(0).unsqueeze(0)
            w_v_input = w_v.unsqueeze(0).unsqueeze(0)

        w_s_padded = F.pad(
            w_s_input,
            (padding_amount, padding_amount, padding_amount, padding_amount),
            mode='circular',
        )
        x_s_filter = F.conv2d(
            w_s_padded,
            kernel.unsqueeze(0).unsqueeze(0),
        ).squeeze()
        
        w_v_padded = F.pad(
            w_v_input,
            (padding_amount, padding_amount, padding_amount, padding_amount),
            mode='circular',
        )
        x_v_filter = F.conv2d(
            w_v_padded,
            kernel.unsqueeze(0).unsqueeze(0),
        ).squeeze()

    else:
        # Transform the padding field to w-domain
        val_s = (1 + alpha_s) / (1 + alpha_s * torch.exp(2 * n_s * (1 - padding)))
        val_v = -(1 + alpha_v) / (1 + alpha_v * torch.exp(2 * n_v * (1 + padding)))
        
        # Required padded size
        target_h, target_w = Nx + 2 * padding_amount, Ny + 2 * padding_amount
        
        # Center crop the background if it's larger than needed
        src_h, src_w = padding.shape[-2:]
        start_h = (src_h - target_h) // 2
        start_w = (src_w - target_w) // 2
        
        # Extract background canvas
        # Padding tensor is likely (H_pad, W_pad) or maybe (1, H_pad, W_pad)
        # We need to broadcast it to batch size if in batch mode
        
        # Shared padding for all items in batch
        bg_s = val_s[start_h:start_h + target_h, start_w:start_w + target_w]
        bg_v = val_v[start_h:start_h + target_h, start_w:start_w + target_w]
        
        if batch_processing:
            Nbatch = x.shape[0]
            # Repeat for batch: (B, H, W)
            w_s_padded = bg_s.unsqueeze(0).repeat(Nbatch, 1, 1).clone()
            w_v_padded = bg_v.unsqueeze(0).repeat(Nbatch, 1, 1).clone()
        else:
            w_s_padded = bg_s.clone()
            w_v_padded = bg_v.clone()
            
        # Insert the active region w_s/w_v into the center
        # Using in-place assignment which is tracked by autograd
        if batch_processing:
            w_s_padded[:, padding_amount:padding_amount + Nx, padding_amount:padding_amount + Ny] = w_s
            w_v_padded[:, padding_amount:padding_amount + Nx, padding_amount:padding_amount + Ny] = w_v
            
            # Conv2d expects (B, C, H, W) -> (B, 1, H, W)
            w_s_in = w_s_padded.unsqueeze(1)
            w_v_in = w_v_padded.unsqueeze(1)
        else:
            w_s_padded[padding_amount:padding_amount + Nx, padding_amount:padding_amount + Ny] = w_s
            w_v_padded[padding_amount:padding_amount + Nx, padding_amount:padding_amount + Ny] = w_v
            
            # (1, 1, H, W)
            w_s_in = w_s_padded.unsqueeze(0).unsqueeze(0)
            w_v_in = w_v_padded.unsqueeze(0).unsqueeze(0)

        x_s_filter = F.conv2d(
            w_s_in,
            kernel.unsqueeze(0).unsqueeze(0),
        ).squeeze()

        x_v_filter = F.conv2d(
            w_v_in,
            kernel.unsqueeze(0).unsqueeze(0),
        ).squeeze()
    
    x_s_proj = 1 - torch.exp(-beta_proj * x_s_filter) + x_s_filter * torch.exp(-beta_proj)
    x_v_proj = -1 + torch.exp(beta_proj * x_v_filter) + x_v_filter * torch.exp(-beta_proj)

    x_proj = (x_s_proj + x_v_proj + 1) / 2

    return x_proj

class conditional_generator:
    def __init__(
        self,
        Nx,
		Ny,
		symmetry,
		periodic,
		padding,
        mfs,
        maxiter=100,
        cuda_ind=0,
    ):
        self.device = 'cuda:' + str(cuda_ind) if torch.cuda.is_available() else 'cpu'

        self.Nx = Nx
        self.Ny = Ny
        self.symmetry = symmetry
        self.periodic = periodic
        if padding is not None:
            self.padding = torch.tensor(padding, device=self.device, dtype=torch.float64)
        else:
            self.padding = padding
        self.mfs = mfs
        self.maxiter = maxiter

        # Get Number of Independent Parameters
        if symmetry == 0:
            self.Ndim = Nx*Ny
           
        elif symmetry == 1:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*Ny)
        
        elif symmetry == 2:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*np.floor(Ny/2 + 0.5))
        
        elif symmetry == 4:
            self.Ndim = int(np.floor(Nx/2 + 0.5)*(np.floor(Nx/2 + 0.5) + 1)/2)

    def constraint_violation(self, x_proj, epsilon=0.15):
        g = 1 - torch.abs(2 * x_proj - 1)
        c_viol = (1 / (self.Nx * self.Ny)) * torch.sum(g)

        return c_viol

    def loss(self, x, tau=0.5):
        x_sym = symmetrize_tensor(
            x,
            self.Nx,
            self.Ny,
            self.symmetry,
            self.device,
        )
        x_proj = two_phase_projection(
            x_sym,
            self.Nx,
            self.Ny,
            self.symmetry,
            self.periodic,
            self.padding,
            self.mfs,
            self.device,
        )
        c_viol = self.constraint_violation(x_proj)

        loss = -(1 / (self.Nx * self.Ny)) * torch.sum(self.x_reward * x_proj) + tau * c_viol

        return loss
    
    def ADAM(
        self,
        x_reward,
        beta_ADAM1=0.9,
        beta_ADAM2=0.999,
        eta_ADAM=0.1,
    ):
        # Functional approach: x_reward is the parameter we differentiate wrt
        # x is the hidden variable initiated from x_reward.
        batch_processing = x_reward.ndim == 2
        
        # Symmetrize reward (this will track gradients from x_reward)
        self.x_reward = symmetrize_tensor(x_reward, self.Nx, self.Ny, self.symmetry, self.device)
        
        if batch_processing:
            self.norm = torch.linalg.vector_norm(self.x_reward.reshape(-1, self.Nx * self.Ny), dim=1, keepdim=True)
            self.x_reward = self.x_reward / self.norm.unsqueeze(-1)
        else:
            self.norm = torch.linalg.vector_norm(self.x_reward.reshape(-1))
            self.x_reward = self.x_reward / self.norm
    
        # Initialize x from x_reward (detached to start optimization)
        # Note: We do NOT use requires_grad_(True) because torch.func.grad handles it.
        x = x_reward.clone().detach() / self.norm

        # Functional gradient function
        grad_fn = torch.func.grad(self.loss)
            
        jac_mean = torch.zeros_like(x)
        jac_var = torch.zeros_like(x)
        adam_iter = 0
            
        while True:
            adam_iter += 1
            
            # Loss
            loss_val = self.loss(x)
            #print(loss_val)
            
            # Compute Gradient using functional API (compatible with jacrev)
            # This computes d(loss)/dx at x
            jac = grad_fn(x)

            if adam_iter > self.maxiter:
                break
                
            # Update Average Gradients (Functional updates)
            jac_mean = beta_ADAM1*jac_mean + (1 - beta_ADAM1)*jac
            jac_var = beta_ADAM2*jac_var + (1 - beta_ADAM2)*jac**2
            
            # Unbias Average Gradients
            jac_mean_unbiased = jac_mean/(1 - beta_ADAM1**adam_iter)
            jac_var_unbiased = jac_var/(1 - beta_ADAM2**adam_iter)
            
            # Update Variables (Functional update)
            x = x - eta_ADAM*jac_mean_unbiased/(torch.sqrt(jac_var_unbiased) + 1e-8)
    
        return x
    
    def generate_near_binary(self, x_reward):
        x_opt = self.ADAM(x_reward)
        x_sym = symmetrize_tensor(
            x_opt,
            self.Nx,
            self.Ny,
            self.symmetry,
            self.device,
        )
        x_near_binary = two_phase_projection(
            x_sym,
            self.Nx,
            self.Ny,
            self.symmetry,
            self.periodic,
            self.padding,
            self.mfs,
            self.device,
        ).reshape(-1)

        return x_near_binary, x_near_binary.detach()

    def generate_near_binary_no_grad(self, x_reward):
        # Run optimization without tracking gradients w.r.t x_reward to save memory
        # We detach x_reward so the optimization graph is not built
        x_reward_detached = x_reward.detach()
        x_opt = self.ADAM(x_reward_detached)
        x_sym = symmetrize_tensor(
            x_opt,
            self.Nx,
            self.Ny,
            self.symmetry,
            self.device,
        )
        x_near_binary = two_phase_projection(
            x_sym,
            self.Nx,
            self.Ny,
            self.symmetry,
            self.periodic,
            self.padding,
            self.mfs,
            self.device,
        ).reshape(x_reward.shape[0], -1)

        # Return detached result (though it should already be detached effectively)
        return x_near_binary.detach()