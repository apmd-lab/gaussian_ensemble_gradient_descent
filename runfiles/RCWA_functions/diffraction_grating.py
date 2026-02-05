import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/apmd/minseokhwan/gaussian_ensemble_gradient_descent/runfiles')

import numpy as np
import torcwa
import torch
import gc
import time
from itertools import product
import util.read_mat_data as rmd

class custom_objective:
    def __init__(self,
                 mat_background,
                 mat_multilayer,
                 Nthreads,
                 minimax=False,
                 cuda_ind=0,
                 ):
                 
        torch.set_warn_always(False)
        torch.set_num_threads(Nthreads)
        torch.set_num_interop_threads(Nthreads)
        self.device = 'cuda:' + str(cuda_ind) if torch.cuda.is_available() else 'cpu'
        torcwa.rcwa_geo.device = self.device
        
        self.minimax = minimax
        
        # User-provided multilayer quantities
        self.mat_background = mat_background
        self.mat_multilayer = mat_multilayer

    def set_accuracy(self, n_harmonic):
        self.n_harmonic = torch.tensor(n_harmonic, dtype=torch.int, device=self.device)

    def set_geometry(self, Nx, Ny, period, thickness):
        self.Nx = Nx
        self.Ny = Ny
        self.period = period
        self.thickness = torch.tensor(thickness, dtype=torch.float64, device=self.device)
        
        torcwa.rcwa_geo.dtype = torch.float64
        torcwa.rcwa_geo.grid()
        torcwa.rcwa_geo.edge_sharpness = 1000.
    
    def set_source(self, freq=None, lam=None, angle_inc=None, k_inc=None):
        angle_inc_temp = np.zeros(2)
        if angle_inc is not None:
            angle_inc_temp[:] = angle_inc
        elif k_inc is not None:
            if k_inc[0] == 0 and k_inc[1] == 0:
                angle_inc_temp[:] = 0
            else:
                angle_inc_temp[1] = np.arctan(k_inc[1]/k_inc[0])
                if np.sqrt(np.sum(k_inc**2)) > freq:
                    angle_inc_temp[0] = np.nan
                else:
                    if k_inc[0] == 0:
                        angle_inc_temp[0] = np.arcsin(k_inc[1]/(freq*np.sin(angle_inc_temp[1])))
                    else:
                        angle_inc_temp[0] = np.arcsin(k_inc[0]/(freq*np.cos(angle_inc_temp[1])))
                
        self.angle_inc = torch.tensor(angle_inc_temp, dtype=torch.float64, device=self.device)
        
        if lam is not None:
            self.n_freq = lam.size
            self.freq = torch.tensor(1/lam, dtype=torch.float64, device=self.device)
        elif freq is not None:
            self.n_freq = freq.size
            self.freq = torch.tensor(freq, dtype=torch.float64, device=self.device)
        
        # Load Material Refractive Indices
        mat_type = list(set(np.hstack((self.mat_multilayer, self.mat_background))))
        if freq is None:
            raw_wavelength, self.mat_dict = rmd.load_all(1e3*lam, 'n_k', mat_type)
        elif lam is None:
            raw_wavelength, self.mat_dict = rmd.load_all(1e3/freq, 'n_k', mat_type)
    
    def get_transmission_cost(self, density, t_array, get_grad):
        torcwa.rcwa_geo.Lx = self.period[0]
        torcwa.rcwa_geo.Ly = self.period[1]
        torcwa.rcwa_geo.nx = self.Nx
        torcwa.rcwa_geo.ny = self.Ny
        
        for nf in range(self.n_freq):
            sim = torcwa.rcwa(freq=self.freq[nf], order=self.n_harmonic, L=self.period, dtype=torch.complex128, device=self.device)
            sim.add_input_layer(eps=self.mat_dict[self.mat_background[0]][nf]**2)
            sim.add_output_layer(eps=self.mat_dict[self.mat_background[1]][nf]**2)
            if torch.sum(torch.isnan(self.angle_inc)) > 0:
                return 0
            sim.set_incident_angle(inc_ang=self.angle_inc[0], azi_ang=self.angle_inc[1])
            
            delta_n = self.mat_dict[self.mat_multilayer[0]][nf] - self.mat_dict[self.mat_background[0]][nf]
            eps = (delta_n*density + self.mat_dict[self.mat_background[0]][nf])**2
            sim.add_layer(thickness=self.thickness, eps=eps)
            
            sim.solve_global_smatrix()
            #sim.source_planewave(amplitude=[1.0,0.0], direction='forward')
            
            t_array[0] = sim.S_parameters(orders=[0,1], direction='forward', port='transmission', polarization='xx', ref_order=[0,0])
            t_array[1] = sim.S_parameters(orders=[0,-1], direction='forward', port='transmission', polarization='yy', ref_order=[0,0])
        
        if get_grad:
            cost = -torch.mean(torch.abs(t_array)**2)
        else:
            if self.minimax:
                cost = -torch.min(torch.abs(t_array)**2)
            else:
                cost = -torch.mean(torch.abs(t_array)**2)
        
        return cost

    def get_cost(self, x, get_grad=False):
        x = torch.tensor(x.reshape(self.Nx, self.Ny), dtype=torch.float64, device=self.device)
        
        if get_grad:
            x.requires_grad_(True)

            t_temp = torch.zeros(2, dtype=torch.complex128, device=self.device)
            cost = self.get_transmission_cost(x, t_temp, get_grad)
            
            cost.backward()
            jac = x.grad.detach().cpu().numpy()
            x.grad = None
        else:
            t_temp = torch.zeros(2, dtype=torch.complex128, device=self.device)
            cost = self.get_transmission_cost(x, t_temp, get_grad)
        
        gc.collect()
        
        if get_grad:
            return cost.detach().cpu().numpy(), jac.reshape(-1)
        else:
            return cost.detach().cpu().numpy()

    def get_transmission_and_fields(self, x):
        density = torch.tensor(x.reshape(self.Nx, self.Ny), dtype=torch.float64, device=self.device)
        
        torcwa.rcwa_geo.Lx = self.period[0]
        torcwa.rcwa_geo.Ly = self.period[1]
        torcwa.rcwa_geo.nx = self.Nx
        torcwa.rcwa_geo.ny = self.Ny
        
        sim = torcwa.rcwa(freq=self.freq[0], order=self.n_harmonic, L=self.period, dtype=torch.complex128, device=self.device)
        sim.add_input_layer(eps=self.mat_dict[self.mat_background[0]][0]**2)
        sim.add_output_layer(eps=self.mat_dict[self.mat_background[1]][0]**2)
        if torch.sum(torch.isnan(self.angle_inc)) > 0:
            return 0
        sim.set_incident_angle(inc_ang=self.angle_inc[0], azi_ang=self.angle_inc[1])
        
        delta_n = self.mat_dict[self.mat_multilayer[0]][0] - self.mat_dict[self.mat_background[0]][0]
        eps = (delta_n*density + self.mat_dict[self.mat_background[0]][0])**2
        sim.add_layer(thickness=self.thickness, eps=eps)
        
        sim.solve_global_smatrix()
        
        txx = sim.S_parameters(orders=[0,1], direction='forward', port='transmission', polarization='xx', ref_order=[0,0])
        tyy = sim.S_parameters(orders=[0,-1], direction='forward', port='transmission', polarization='yy', ref_order=[0,0])
        
        Txx = (torch.abs(txx)**2).item()
        Tyy = (torch.abs(tyy)**2).item()
        
        z_field = torch.linspace(-1.0, 4.0, 501, device=self.device)
        
        sim.source_planewave(amplitude=[1.0,0.0], direction='forward')
        [Exx, Exy, Exz], [Hxx, Hxy, Hxz] = sim.field_yz(torcwa.rcwa_geo.y, z_field, self.period[0]/2)
        
        sim.source_planewave(amplitude=[0.0,1.0], direction='forward')
        [Eyx, Eyy, Eyz], [Hyx, Hyy, Hyz] = sim.field_yz(torcwa.rcwa_geo.y, z_field, self.period[0]/2)
        
        y_field = torcwa.rcwa_geo.y.detach().cpu().numpy()
        z_field = z_field.detach().cpu().numpy()
        ReExx = torch.real(Exx).detach().cpu().numpy()
        ReEyy = torch.real(Eyy).detach().cpu().numpy()
        
        return Txx, Tyy, y_field, z_field, ReExx, ReEyy