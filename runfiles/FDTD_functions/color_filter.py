import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-15])

import numpy as np
import autograd.numpy as npa
from autograd import grad
import torch
import torcwa
import time
from itertools import product
import util.read_mat_data as rmd
import util.read_txt as txt

sys.path.append('/opt/lumerical/v231/api/python/')
import lumapi

nm = 1e-9
um = 1e-6
c = 299792458

class custom_objective:
    def __init__(self,
                 lam, # in m
                 design_dim, # in m
                 Nx,
                 Ny,
                 mat_pattern,
                 mat_void,
                 mat_substrate,
                 d_mesh,
                 ciexy_target,
                 Nthreads,
                 n_harmonic,
                 ):
                 
        # Source
        self.lam = lam
        self.freq = c/self.lam
        self.N_freq = freq.size
        
        # Geometries
        self.dx_design, self.dy_design, self.dz_design = design_dim
        self.dx_pix = self.dx_design/Nx
        self.dy_pix = self.dy_design/Ny
        self.Nx = Nx
        self.Ny = Ny

        # Materials
        mat_type = np.array([mat_pattern,mat_void,mat_substrate])
        raw_wavelength, mat_dict = rmd.load_all(lam, 'n_k', mat_type)
        self.n_pattern = np.mean(np.real(mat_dict[mat_pattern]))
        self.n_void = np.mean(np.real(mat_dict[mat_void]))
        self.n_substrate = np.mean(np.real(mat_dict[mat_substrate]))

        # FDTD Settings
        self.d_mesh = d_mesh
        
        # RCWA Settings
        torch.set_warn_always(False)
        torch.set_num_threads(Nthreads)
        torch.set_num_interop_threads(Nthreads)
        self.device = 'cpu'
        torcwa.rcwa_geo.device = self.device
        
        self.n_harmonic = torch.tensor(n_harmonic, dtype=torch.int, device=self.device)

        # Color
        self.ciexy_target = ciexy_target
        
        raw = txt.read_txt(directory + '/CIE_XYZ_Color_Matching_Functions(2deg)')
        self.x_bar = np.zeros((3, self.N_freq))
        for i in range(3):
            self.x_bar[i,:] = np.interp(lam_out, raw[:,0], raw[:,i+1])
            
        raw = txt.read_txt(directory + '/D65_Intensity_Spectrum')
        self.illum = np.interp(lam_out, raw[:,0], raw[:,1])

        # Create FDTD Simulation File
        self.make_FDTD_sim()

    def set_accuracy(self, simulator):
        self.simulator = simulator
    
    def make_FDTD_sim(self):
        self.fdtd = lumapi.FDTD(hide=True)
        self.fdtd.eval('save("%s");' %(directory[:-15] + '/color_filter.fsp'))
            
        # Simulation Region
        sim = self.fdtd.addfdtd()
        sim['dimension'] = '3D'
        sim['x min'] = -self.dx_design/2
        sim['x max'] = self.dx_design/2
        sim['y min'] = -self.dy_design/2
        sim['y max'] = self.dy_design/2
        sim['z min'] = -self.dz_design/2 - 1.5*np.max(self.lam)
        sim['z max'] = self.dz_design/2 + 1.5*np.max(self.lam)
        
        sim['mesh accuracy'] = 3
        sim['force symmetric x mesh'] = 1
        sim['force symmetric y mesh'] = 1
        sim['force symmetric z mesh'] = 1
        sim['x min bc'] = 'Anti-Symmetric'
        sim['x max bc'] = 'Anti-Symmetric'
        sim['y min bc'] = 'Symmetric'
        sim['y min bc'] = 'Symmetric'
        sim['z min bc'] = 'PML'
        sim['pml layers'] = 64
        sim['index'] = self.n_void
            
        # Output Monitor
        field_out = self.fdtd.addpower()
        field_out['name'] = 'output_monitor'

        field_out['monitor type'] = '2D Z-normal'
        field_out['x'] = 0.0
        field_out['y'] = 0.0
        field_out['z'] = self.dz_design/2 + np.max(self.lam)
        field_out['x span'] = self.dx_design
        field_out['y span'] = self.dy_design
        
        # Design Region Monitor
        field_design = self.fdtd.addpower()
        field_design['name'] = 'design_region_monitor'
        
        field_design['monitor type'] = '3D'
        field_design['x'] = 0.0
        field_design['y'] = 0.0
        field_design['z'] = 0.0
        field_design['x span'] = self.x_design
        field_design['y span'] = self.y_design
        field_design['z span'] = self.z_design
            
        # Sources
        src_fwd = self.fdtd.addplane()
        src_fwd['name'] = 'source_fwd'
        src_fwd['direction'] = 'Forward'
        src_fwd['injection axis'] = 'z-axis'
        src_fwd['wavelength start'] = self.lam[0]
        src_fwd['wavelength stop'] = self.lam[-1]
        
        src_fwd['x'] = 0.0
        src_fwd['y'] = 0.0
        src_fwd['z'] = -self.dz_design/2 - np.max(self.lam)
        src_fwd['x span'] = self.dx_design
        src_fwd['y span'] = self.dy_design
            
        src_adj = self.fdtd.addplane()
        src_adj['name'] = 'source_adj'
        src_adj['direction'] = 'Backward'
        src_adj['injection axis'] = 'z-axis'
        src_adj['wavelength start'] = self.lam[0]
        src_adj['wavelength stop'] = self.lam[-1]
        
        src_adj['x'] = 0.0
        src_adj['y'] = 0.0
        src_adj['z'] = self.dz_design/2 + np.max(self.lam)
        src_adj['x span'] = self.dx_design
        src_adj['y span'] = self.dy_design

        # Mesh
        mesh = self.fdtd.addmesh()
        mesh['x'] = 0.0
        mesh['y'] = 0.0
        mesh['z'] = 0.0
        mesh['x span'] = self.x_design
        mesh['y span'] = self.y_design
        mesh['z span'] = self.z_design
        mesh['dx'] = self.d_mesh
        mesh['dy'] = self.d_mesh
        mesh['dz'] = self.d_mesh

        # Substrate
        sub = self.fdtd.addrect();
        sub['name'] = 'substrate'
        sub['index'] = self.n_substrate
        sub['x'] = 0.0
        sub['x span'] = self.dx_design
        sub['y'] = 0.0
        sub['y span'] = self.dy_design
        sub['z min'] = -self.dz_design/2 - np.max(self.lam)
        sub['z max'] = -self.dz_design/2
    
        # Generate Pattern
        for nx in range(self.Nx):
            for ny in range(self.Ny):
                rect = self.fdtd.addrect()
                rect['name'] = 'uc_' + str(nx) + '_' + str(ny)
                
                rect['x min'] = -self.dx_design/2 + nx*self.dx_pix
                rect['x max'] = -self.dx_design/2 + (nx + 1)*self.dx_pix
                rect['y min'] = -self.dy_design/2 + ny*self.dy_pix
                rect['y max'] = -self.dy_design/2 + (ny + 1)*self.dy_pix
                rect['z'] = 0.0
                rect['z span'] = self.dz_design
        
        self.fdtd.eval('save;')
    
    def update_design(self, density):
        #self.fdtd = lumapi.FDTD(directory[:-15] + '/waveguide_crossing' + str(self.dimension) + 'D.fsp', hide=True)
        self.fdtd.switchtolayout()
        
        # Update Design
        nLow = self.n_void
        nHigh = self.n_pattern
        dn = nHigh - nLow
        RI = np.real(nLow + dn*density)
        
        for nx in range(self.Nx):
            for ny in range(self.Ny):
                self.fdtd.eval('setnamed("uc_' + str(nx) + '_' + str(ny) + '","index",' + str(RI[nx,ny]) + ');')
        
        self.fdtd.eval('save;')
    
    def get_RCWA_cost(self, density, period, t_array):
        delta_n = self.n_pattern - self.n_void
        
        for nf in range(self.N_freq):
            sim = torcwa.rcwa(freq=self.freq[nf], order=self.n_harmonic, L=period, dtype=torch.complex128, device=self.device)
            sim.add_input_layer(eps=self.n_substrate**2)
            sim.add_output_layer(eps=self.n_void**2)
            sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)

            eps = (delta_n*density + self.n_void)**2
            sim.add_layer(thickness=self.dz_design, eps=eps)
            
            sim.solve_global_smatrix()
            
            t_array[nf] = sim.S_parameters(orders=[0,0], direction='forward', port='transmission', polarization='xx', ref_order=[0,0])

        T = torch.abs(t_array)**2
        
        # complete rest of the cost evaluation code
                
        cost = -torch.mean(torch.abs(t_array)**2)
        cost_exp = -torch.exp(-5*cost)
        
        return cost_exp
    
    def get_cost(self, x, get_grad=False):
        if self.simulator == 'FDTD':
            self.update_design(x)
        
            # Run Forward Simulation
            self.fdtd.switchtolayout()
            self.fdtd.eval('setnamed("source_fwd", "enabled", 1);')
            self.fdtd.eval('setnamed("source_adj", "enabled", 0);')
            self.fdtd.run()
            
            # Output Monitor Data
            f_out = np.squeeze(self.fdtd.getdata("output_monitor", "f"))
            lam_out = c/f_out/nm
            T_out = np.squeeze(self.fdtd.getdata("output_monitor", "T"))
    
            # Design Region Field Data
            x_design = np.squeeze(self.fdtd.getdata("design_region_monitor","x"))
            y_design = np.squeeze(self.fdtd.getdata("design_region_monitor","y"))
            z_design = np.squeeze(self.fdtd.getdata("design_region_monitor","z"))
            E_fwd = np.zeros((3, x_design, y_design, z_design, self.N_freq)).astype(np.complex128)
            E_fwd[0,:,:,:,:] = np.squeeze(self.fdtd.getdata("design_region_monitor","Ex"))
            E_fwd[1,:,:,:,:] = np.squeeze(self.fdtd.getdata("design_region_monitor","Ey"))
            E_fwd[2,:,:,:,:] = np.squeeze(self.fdtd.getdata("design_region_monitor","Ez"))
            
            if get_grad:
                # Run Adjoint Simulation
                self.fdtd.switchtolayout()
                self.fdtd.eval('setnamed("source_fwd", "enabled", 0);')
                self.fdtd.eval('setnamed("source_adj", "enabled", 1);')
                self.fdtd.run()
                
                # Design Region Field Data
                x_design = np.squeeze(self.fdtd.getdata("design_region_monitor","x"))
                y_design = np.squeeze(self.fdtd.getdata("design_region_monitor","y"))
                z_design = np.squeeze(self.fdtd.getdata("design_region_monitor","z"))
                E_adj = np.zeros((3, x_design, y_design, z_design, self.N_freq)).astype(np.complex128)
                E_adj[0,:,:,:,:] = np.squeeze(self.fdtd.getdata("design_region_monitor","Ex"))
                E_adj[1,:,:,:,:] = np.squeeze(self.fdtd.getdata("design_region_monitor","Ey"))
                E_adj[2,:,:,:,:] = np.squeeze(self.fdtd.getdata("design_region_monitor","Ez"))
            
            # Color
            def CIExy(T_out, x_bar, illum, ciexy_target):
                I = npa.diag(illum)
                A = x_bar @ I
                XYZ = A @ T_out
                
                xy = XYZ[:2]/npa.sum(XYZ)
            
                diff = npa.sqrt(npa.sum((xy - ciexy_target)**2))
                
                cost = 2*diff - 1
                cost_exp = -npa.exp(-5*cost)
                
                return cost_exp
            
            dCIExy = grad(CIExy, argnum=0)
    
            T_out = npa.array(T_out)
            x_bar = npa.array(self.x_bar)
            illum = npa.array(self.illum)
            ciexy_target = npa.array(self.ciexy_target)
    
            cost_exp = CIExy(T_out, x_bar, illum, ciexy_target)
            
            if get_grad:
                nLow = self.mat_dict[self.mat_padding][0]
                nHigh = self.mat_dict[self.mat_waveguide][0]
                dn = nHigh - nLow
                RI = np.real(nLow + dn*density)
                
                dedr = 2*RI*dn
                dE_xy = np.sum(E_fwd*E_adj, axis=(0,3))
                jac_spectrum = 2*np.real(dE_xy*dedr[:,:,np.newaxis])
                jac_CIExy = dCIExy(T_out, x_bar, illum, ciexy_target)
                jac_exp = np.sum(jac_CIExy[np.newaxis,np.newaxis,:]*jac_spectrum, axis=-1)
                
                jac_exp_reduced = np.zeros((self.Nx, self.Ny))
                for nx in range(self.Nx):
                    for ny in range(self.Ny):
                        x_mask = (x_design > -dx_design)*(x_design < -dx_design + nx*self.dx_pix)
                        y_mask = (y_design > -dy_design)*(y_design < -dy_design + ny*self.dy_pix)
                        jac_temp_x = jac_exp[x_mask,:]
                        jac_temp_xy = jac_temp_x[:,y_mask]
                        jac_exp_reduced[nx,ny] = np.sum(jac_temp_xy)
            
            if get_grad:
                return cost_exp, jac_exp_reduced
            else:
                return cost_exp
        
        elif self.simulator == 'RCWA':
            torcwa.rcwa_geo.dtype = torch.float64
            torcwa.rcwa_geo.grid()
            torcwa.rcwa_geo.edge_sharpness = 1000.
            
            freq = torch.tensor(1/self.lam/um, dtype=torch.float64, device=self.device)
        
            torcwa.rcwa_geo.Lx = self.dx_design/um
            torcwa.rcwa_geo.Ly = self.dy_design/um
            torcwa.rcwa_geo.nx = self.Nx
            torcwa.rcwa_geo.ny = self.Ny
            
            period = torch.tensor([self.dx_design/um,self.dy_design/um], dtype=torch.float64, device=self.device)
            
            

    def get_cost(self, x, get_grad=False):
        x = torch.tensor(x.reshape(self.Nx, self.Ny), dtype=torch.float64, device=self.device)
        
        if get_grad:
            x.requires_grad_(True)

            t_temp = torch.zeros(2, dtype=torch.complex128, device=self.device)
            cost = self.get_transmission_cost(x, t_temp)
            
            cost.backward()
            jac = x.grad.detach().cpu().numpy()
            x.grad = None
        else:
            t_temp = torch.zeros(2, dtype=torch.complex128, device=self.device)
            cost = self.get_transmission_cost(x, t_temp)
        
        gc.collect()
        
        if get_grad:
            return cost.detach().cpu().numpy(), jac.reshape(-1)
        else:
            return cost.detach().cpu().numpy()