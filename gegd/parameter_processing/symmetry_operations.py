import numpy as np

def symmetrize(x, symmetry, Nx, Ny):
    if symmetry == 0:
        x_sym = x.reshape(Nx, Ny)
    
    elif symmetry == 1:
        x_half = x.reshape(int(np.floor(Nx/2 + 0.5)), Ny)
        x_sym = np.zeros((Nx, Ny))
        
        if Nx % 2 == 0:
            x_sym[:int(Nx/2),:] = x_half
            x_sym[int(Nx/2):,:] = np.flip(x_half, axis=0)
        
        else:
            x_sym[:int((Nx+1)/2),:] = x_half
            x_sym[int((Nx+1)/2):,:] = np.flip(x_half[:-1,:], axis=0)
    
    elif symmetry == 2:
        x_quart = x.reshape(int(np.floor(Nx/2 + 0.5)), int(np.floor(Ny/2 + 0.5)))
        x_sym = np.zeros((Nx, Ny))
        
        if Nx % 2 == 0 and Ny % 2 == 0:
            x_sym[:int(Nx/2),:int(Ny/2)] = x_quart
            x_sym[int(Nx/2):,:int(Ny/2)] = np.flip(x_quart, axis=0)
            x_sym[:int(Nx/2),int(Ny/2):] = np.flip(x_quart, axis=1)
            x_sym[int(Nx/2):,int(Ny/2):] = np.flip(np.flip(x_quart, axis=0), axis=1)
        
        elif Nx % 2 == 1 and Ny % 2 == 0:
            x_sym[:int((Nx+1)/2),:int(Ny/2)] = x_quart
            x_sym[int((Nx+1)/2):,:int(Ny/2)] = np.flip(x_quart[:-1,:], axis=0)
            x_sym[:int((Nx+1)/2),int(Ny/2):] = np.flip(x_quart, axis=1)
            x_sym[int((Nx+1)/2):,int(Ny/2):] = np.flip(np.flip(x_quart[:-1,:], axis=0), axis=1)
        
        elif Nx % 2 == 0 and Ny % 2 == 1:
            x_sym[:int(Nx/2),:int((Ny+1)/2)] = x_quart
            x_sym[int(Nx/2):,:int((Ny+1)/2)] = np.flip(x_quart, axis=0)
            x_sym[:int(Nx/2),int((Ny+1)/2):] = np.flip(x_quart[:,:-1], axis=1)
            x_sym[int(Nx/2):,int((Ny+1)/2):] = np.flip(np.flip(x_quart[:,:-1], axis=0), axis=1)
        
        elif Nx % 2 == 1 and Ny % 2 == 1:
            x_sym[:int((Nx+1)/2),:int((Ny+1)/2)] = x_quart
            x_sym[int((Nx+1)/2):,:int((Ny+1)/2)] = np.flip(x_quart[:-1,:], axis=0)
            x_sym[:int((Nx+1)/2),int((Ny+1)/2):] = np.flip(x_quart[:,:-1], axis=1)
            x_sym[int((Nx+1)/2):,int((Ny+1)/2):] = np.flip(np.flip(x_quart[:-1,:-1], axis=0), axis=1)
    
    elif symmetry == 4:
        assert Nx == Ny, "The pixel numbers in the X and Y directions must be equal under D4 symmetry"
        
        x = x.reshape(-1)            
        if Nx % 2 == 0:
            x_quart = np.zeros((int(Nx/2), int(Ny/2)))
            x_quart[np.triu_indices(int(Nx/2))] = x
            x_quart = x_quart + x_quart.T - np.diag(np.diag(x_quart))
            
            x_sym = np.zeros((Nx, Ny))
            x_sym[:int(Nx/2),:int(Ny/2)] = x_quart
            x_sym[int(Nx/2):,:int(Ny/2)] = np.rot90(x_quart)
            x_sym[int(Nx/2):,int(Ny/2):] = np.rot90(x_quart, k=2)
            x_sym[:int(Nx/2),int(Ny/2):] = np.rot90(x_quart, k=-1)
        
        else:
            x_quart = np.zeros((int((Nx+1)/2), int((Ny+1)/2)))
            x_quart[np.triu_indices(int((Nx+1)/2))] = x
            x_quart = x_quart + x_quart.T - np.diag(np.diag(x_quart))
            
            x_sym = np.zeros((Nx, Ny))
            x_sym[:int((Nx+1)/2),:int((Ny+1)/2)] = x_quart
            x_sym[int((Nx+1)/2):,:int((Ny+1)/2)] = np.rot90(x_quart)[1:,:]
            x_sym[int((Nx+1)/2):,int((Ny+1)/2):] = np.rot90(x_quart, k=2)[1:,1:]
            x_sym[:int((Nx+1)/2),int((Ny+1)/2):] = np.rot90(x_quart, k=-1)[:,1:]
    
    return x_sym

def symmetrize_jacobian(jac, symmetry, Nx, Ny):
    if symmetry == 0:
        jac_sym = symmetrize(jac, symmetry, Nx, Ny)
    
    elif symmetry == 1:
        if Nx % 2 == 0:
            scaling_coeff = (1/2)*np.ones((int(Nx/2), Ny))
            
        else:
            scaling_coeff = (1/2)*np.ones((int((Nx+1)/2), Ny))
            scaling_coeff[-1,:] = 1
        
        jac *= scaling_coeff.reshape(-1)
        jac_sym = symmetrize(jac, symmetry, Nx, Ny)
    
    elif symmetry == 2:
        if Nx % 2 == 0 and Ny % 2 == 0:
            scaling_coeff = (1/4)*np.ones((int(Nx/2), int(Ny/2)))
        
        elif Nx % 2 == 1 and Ny % 2 == 0:
            scaling_coeff = (1/4)*np.ones((int((Nx+1)/2), int(Ny/2)))
            scaling_coeff[-1,:] = 1/2
        
        elif Nx % 2 == 0 and Ny % 2 == 1:
            scaling_coeff = (1/4)*np.ones((int(Nx/2), int((Ny+1)/2)))
            scaling_coeff[:,-1] = 1/2
        
        else:
            scaling_coeff = (1/4)*np.ones((int((Nx+1)/2), int((Ny+1)/2)))
            scaling_coeff[-1,:] = 1/2
            scaling_coeff[:,-1] = 1/2
            scaling_coeff[-1,-1] = 1
        
        jac *= scaling_coeff.reshape(-1)
        jac_sym = symmetrize(jac, symmetry, Nx, Ny)
    
    elif symmetry == 4:
        if Nx % 2 == 0:
            scaling_coeff = (1/8)*np.ones((int(Nx/2), int(Ny/2)))
            scaling_coeff += (1/8)*np.identity(int(Nx/2))
            scaling_coeff = scaling_coeff[np.triu_indices(int(Nx/2))]
        
        else:
            scaling_coeff = (1/8)*np.ones((int((Nx+1)/2), int((Ny+1)/2)))
            scaling_coeff += (1/8)*np.identity(int((Nx+1)/2))
            scaling_coeff[:,-1] = 1/4
            scaling_coeff[-1,-1] = 1
            scaling_coeff = scaling_coeff[np.triu_indices(int((Nx+1)/2))]
        
        jac *= scaling_coeff
        jac_sym = symmetrize(jac, symmetry, Nx, Ny)
    
    return jac_sym

def desymmetrize(x_sym, symmetry, Nx, Ny):
    if symmetry == 0:
        x = x_sym.reshape(-1)
    
    elif symmetry == 1:
        x_sym = x_sym.reshape(Nx, Ny)
        if Nx % 2 == 0:
            x_half = x_sym[:int(Nx/2),:]
        else:
            x_half = x_sym[:int((Nx+1)/2),:]
            
        x = x_half.reshape(-1)
    
    elif symmetry == 2:
        x_sym = x_sym.reshape(Nx, Ny)
        if Nx % 2 == 0 and Ny % 2 == 0:
            x_quart = x_sym[:int(Nx/2),:int(Ny/2)]
        elif Nx % 2 == 1 and Ny % 2 == 0:
            x_quart = x_sym[:int((Nx+1)/2),:int(Ny/2)]
        elif Nx % 2 == 0 and Ny % 2 == 1:
            x_quart = x_sym[:int(Nx/2),:int((Ny+1)/2)]
        else:
            x_quart = x_sym[:int((Nx+1)/2),:int((Ny+1)/2)]
        
        x = x_quart.reshape(-1)
    
    elif symmetry == 4:
        x_sym = x_sym.reshape(Nx, Ny)
        if Nx % 2 == 0:
            x_quart = x_sym[:int(Nx/2),:int(Ny/2)]
            x = x_quart[np.triu_indices(int(Nx/2))]
        
        else:
            x_quart = x_sym[:int((Nx+1)/2),:int((Ny+1)/2)]
            x = x_quart[np.triu_indices(int((Nx+1)/2))]
    
    return x
    
def desymmetrize_jacobian(jac_sym, symmetry, Nx, Ny):
    if symmetry == 0:
        jac = jac_sym.copy()
    
    elif symmetry == 1:
        if Nx % 2 == 0:
            jac = jac_sym[:int(Nx/2),:]
            jac += np.flip(jac_sym[int(Nx/2):,:], axis=0)
        
        else:
            jac = jac_sym[:int((Nx+1)/2),:]
            jac[:-1,:] += np.flip(jac_sym[int((Nx+1)/2):,:], axis=0)
    
    elif symmetry == 2:
        if Nx % 2 == 0 and Ny % 2 == 0:
            jac = jac_sym[:int(Nx/2),:int(Ny/2)]
            jac += np.flip(jac_sym[int(Nx/2):,:int(Ny/2)], axis=0)
            jac += np.flip(jac_sym[:int(Nx/2),int(Ny/2):], axis=1)
            jac += np.flip(jac_sym[int(Nx/2):,int(Ny/2):], axis=(0,1))
        
        elif Nx % 2 == 1 and Ny % 2 == 0:
            jac = jac_sym[:int((Nx+1)/2),:int(Ny/2)]
            jac[:-1,:] += np.flip(jac_sym[int((Nx+1)/2):,:int(Ny/2)], axis=0)
            jac += np.flip(jac_sym[:int((Nx+1)/2),int(Ny/2):], axis=1)
            jac[:-1,:] += np.flip(jac_sym[int((Nx+1)/2):,int(Ny/2):], axis=(0,1))
        
        elif Nx % 2 == 0 and Ny % 2 == 1:
            jac = jac_sym[:int(Nx/2),:int((Ny+1)/2)]
            jac += np.flip(jac_sym[int(Nx/2):,:int((Ny+1)/2)], axis=0)
            jac[:,:-1] += np.flip(jac_sym[:int(Nx/2),int((Ny+1)/2):], axis=1)
            jac[:,:-1] += np.flip(jac_sym[int(Nx/2):,int((Ny+1)/2):], axis=(0,1))
        
        else:
            jac = jac_sym[:int((Nx+1)/2),:int((Ny+1)/2)]
            jac[:-1,:] = np.flip(jac_sym[int((Nx+1)/2):,:int((Ny+1)/2)], axis=0)
            jac[:,:-1] = np.flip(jac_sym[:int((Nx+1)/2),int((Ny+1)/2):], axis=1)
            jac[:-1,:-1] = np.flip(jac_sym[int((Nx+1)/2):,int((Ny+1)/2):], axis=(0,1))
    
    elif symmetry == 4:
        if Nx % 2 == 0:
            jac_quart = jac_sym[:int(Nx/2),:int(Ny/2)]
            jac_quart += np.flip(jac_sym[int(Nx/2):,:int(Ny/2)], axis=0)
            jac_quart += np.flip(jac_sym[:int(Nx/2),int(Ny/2):], axis=1)
            jac_quart += np.flip(jac_sym[int(Nx/2):,int(Ny/2):], axis=(0,1))
            jac_quart += jac_quart.T - np.diag(np.diag(jac_quart))
            
            jac = jac_quart[np.triu_indices(int(Nx/2))]

        else:
            jac_quart = jac_sym[:int((Nx+1)/2),:int((Ny+1)/2)]
            jac_quart[:-1,:] = np.flip(jac_sym[int((Nx+1)/2):,:int((Ny+1)/2)], axis=0)
            jac_quart[:,:-1] = np.flip(jac_sym[:int((Nx+1)/2),int((Ny+1)/2):], axis=1)
            jac_quart[:-1,:-1] = np.flip(jac_sym[int((Nx+1)/2):,int((Ny+1)/2):], axis=(0,1))
            jac_quart += jac_quart.T - np.diag(np.diag(jac_quart))
            
            jac = jac_quart[np.triu_indices(int((Nx+1)/2))]
    
    return jac.reshape(-1)