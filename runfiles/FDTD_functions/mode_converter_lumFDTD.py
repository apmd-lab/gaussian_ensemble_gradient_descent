import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-15])

import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator
from itertools import product
import time
import subprocess
import util.read_mat_data as rmd

sys.path.append('/opt/lumerical/v231/api/python/')
import lumapi

nm = 1e-9
c = 299792458

class custom_objective:
    def __init__(self,
                 dimension,
                 lam_tgt, # in m
                 design_dim, # in m
                 waveguide_width, # in m
                 Nx,
                 Ny,
                 mat_padding,
                 mat_waveguide,
                 Nthreads,
                 fsp_suffix,
                 symmetric_bounds=True,
                 ):
                 
        # Source
        self.lam_tgt = lam_tgt
        self.N_freq = 1
        
        # Geometries
        self.dx_design, self.dy_design, self.dz_design = design_dim
        self.dx_pix = self.dx_design/Nx
        self.dy_pix = self.dy_design/Ny
        self.Nx = Nx
        self.Ny = Ny
        self.waveguide_width = waveguide_width
        self.symmetric_bounds = True

        # Materials
        self.mat_padding = mat_padding
        self.mat_waveguide = mat_waveguide

        # Other Simulation Settings
        self.dimension = dimension
        self.Nthreads = Nthreads
        self.fsp_suffix = fsp_suffix

        # Create FDTD Simulation File
        self.make_sim()

    def set_accuracy(self, upsampling_ratio_fdtd):
        # Update Mesh Density
        self.fdtd.switchtolayout()
        
        self.upsampling_ratio_fdtd = upsampling_ratio_fdtd
        d_mesh = self.dx_design/(upsampling_ratio_fdtd*self.Nx)
        
        self.fdtd.eval('setnamed("mesh_design", "dx", ' + str(d_mesh) + ');')
        self.fdtd.eval('setnamed("mesh_design", "dy", ' + str(d_mesh) + ');')
        self.fdtd.eval('setnamed("mesh_design", "dz", ' + str(d_mesh) + ');')
        
        # Update Spectrum Sampling Points
        self.fdtd.eval('setglobalmonitor("frequency points", ' + str(self.N_freq) + ');')
        self.fdtd.eval('setglobalmonitor("use wavelength spacing", true);')
        
        # Load Material Refractive Indices
        mat_type = np.array([self.mat_padding,self.mat_waveguide])
        raw_wavelength, mat_dict = rmd.load_all(np.array([self.lam_tgt/nm]), 'n_k', mat_type)
        self.n_padding = np.mean(np.real(mat_dict[self.mat_padding]))
        self.n_waveguide = np.mean(np.real(mat_dict[self.mat_waveguide]))
        self.fdtd.eval('select("FDTD");')
        self.fdtd.eval('set("index", ' + str(self.n_padding) + ');')
        self.fdtd.eval('select("waveguide_in");')
        self.fdtd.eval('set("index", ' + str(self.n_waveguide) + ');')
        self.fdtd.eval('select("waveguide_out");')
        self.fdtd.eval('set("index", ' + str(self.n_waveguide) + ');')
        
        # Get Target Mode Profile
        self.fdtd.eval('select("source_fwd");')
        self.fdtd.eval('set("mode selection", "user select");')
        self.fdtd.eval('set("selected mode number", 3);') 
        self.fdtd.eval('mode = getresult("source_fwd","mode profile");')
        self.fdtd.eval('Ex = mode.Ex;')
        self.fdtd.eval('Ey = mode.Ey;')
        self.fdtd.eval('Ez = mode.Ez;')
        self.fdtd.eval('Hx = mode.Hx;')
        self.fdtd.eval('Hy = mode.Hy;')
        self.fdtd.eval('Hz = mode.Hz;')
        Ex = self.fdtd.getv('Ex')
        self.E_mode_out = np.zeros((3, Ex.shape[0], Ex.shape[1], Ex.shape[2], self.N_freq)).astype(np.complex128)
        self.E_mode_out[0,:,:,:,:] = self.fdtd.getv('Ex')
        self.E_mode_out[1,:,:,:,:] = self.fdtd.getv('Ey')
        self.E_mode_out[2,:,:,:,:] = self.fdtd.getv('Ez')
        self.H_mode_out = np.zeros((3, Ex.shape[0], Ex.shape[1], Ex.shape[2], self.N_freq)).astype(np.complex128)
        self.H_mode_out[0,:,:,:,:] = self.fdtd.getv('Hx')
        self.H_mode_out[1,:,:,:,:] = self.fdtd.getv('Hy')
        self.H_mode_out[2,:,:,:,:] = self.fdtd.getv('Hz')
        
        self.fdtd.eval('set("mode selection", "fundamental mode");')
    
    def make_sim(self):
        t1 = time.time()
        while True:
            try:
                self.fdtd = lumapi.FDTD(hide=True)
                break
            except:
                t2 = time.time()
                print('Waiting for license (' + str(np.round(t2 - t1)) + 's)', flush=True)
                time.sleep(60)
        self.fdtd.eval('save("%s");' %(directory[:-15] + '/mode_converter_' + self.fsp_suffix + '.fsp'))
            
        # Simulation Region
        if self.dimension == 2:
            sim = self.fdtd.addfdtd()
            sim['dimension'] = '2D'
            sim['x min'] = -self.dx_design/2 - self.lam_tgt/2
            sim['x max'] = self.dx_design/2 + self.lam_tgt/2
            sim['y min'] = -self.dy_design/2 - self.lam_tgt
            sim['y max'] = self.dy_design/2 + self.lam_tgt
            sim['z'] = 0.0
            
            sim['mesh accuracy'] = 3
            sim['force symmetric x mesh'] = 1
            sim['force symmetric y mesh'] = 1
            if self.symmetric_bounds:
                sim['x min bc'] = 'Symmetric'
            else:
                sim['x min bc'] = 'PML'
            sim['y min bc'] = 'PML'
            sim['pml layers'] = 16
             
        if self.dimension == 3:
            sim = self.fdtd.addfdtd()
            sim['dimension'] = '3D'
            sim['x min'] = -self.dx_design/2 - self.lam_tgt/2
            sim['x max'] = self.dx_design/2 + self.lam_tgt/2
            sim['y min'] = -self.dy_design/2 - self.lam_tgt
            sim['y max'] = self.dy_design/2 + self.lam_tgt
            sim['z min'] = -self.dz_design/2 - self.lam_tgt
            sim['z max'] = self.dz_design/2 + self.lam_tgt
            
            sim['mesh accuracy'] = 3
            sim['force symmetric x mesh'] = 1
            sim['force symmetric y mesh'] = 1
            sim['force symmetric z mesh'] = 1
            if self.symmetric_bounds:
                sim['x min bc'] = 'Symmetric'
            else:
                sim['x min bc'] = 'PML'
            sim['y min bc'] = 'PML'
            sim['z min bc'] = 'PML'
            sim['pml layers'] = 16
        
        # Movie Monitor
#        movie = self.fdtd.addmovie()
#        movie['monitor type'] = 3
#        movie['x'] = 0.0
#        movie['y'] = 0.0
#        movie['z'] = 0.0
#        movie['x span'] = self.dx_design + self.lam_tgt
#        movie['y span'] = self.dy_design + 2*self.lam_tgt
#        movie['lock aspect ratio'] = 1
        
        # Output Monitor
        field_out21 = self.fdtd.addpower()
        field_out21['name'] = 'output_monitor21'
        
        if self.dimension == 2:
            field_out21['monitor type'] = 'Linear X'
            field_out21['x'] = 0.0
            field_out21['y'] = self.dy_design/2 + self.lam_tgt/2
            field_out21['z'] = 0.0
            field_out21['x span'] = 2*self.waveguide_width
            
        if self.dimension == 3:
            field_out21['monitor type'] = '2D Y-normal'
            field_out21['x'] = 0.0
            field_out21['y'] = self.dy_design/2 + self.lam_tgt/2
            field_out21['z'] = 0.0
            field_out21['x span'] = 2*self.waveguide_width
            field_out21['z span'] = 2*self.dz_design
        
        field_out11 = self.fdtd.addpower()
        field_out11['name'] = 'output_monitor11'
        
        if self.dimension == 2:
            field_out11['monitor type'] = 'Linear X'
            field_out11['x'] = 0.0
            field_out11['y'] = -self.dy_design/2 - 0.6*self.lam_tgt
            field_out11['z'] = 0.0
            field_out11['x span'] = 2*self.waveguide_width
        
        if self.dimension == 3:
            field_out11['monitor type'] = '2D Y-normal'
            field_out11['x'] = 0.0
            field_out11['y'] = -self.dy_design/2 - 0.6*self.lam_tgt
            field_out11['z'] = 0.0
            field_out11['x span'] = 2*self.waveguide_width
            field_out11['z span'] = 2*self.dz_design
        
        mode_out21 = self.fdtd.addmodeexpansion()
        mode_out21['name'] = 'mode_monitor21'
        mode_out21['mode selection'] = 'user select'
        self.fdtd.eval('set("selected mode numbers", 3);')
        self.fdtd.eval('setexpansion("input", "output_monitor21");')

        mode_out21['monitor type'] = '2D Y-normal'
        mode_out21['x'] = 0.0
        mode_out21['y'] = self.dy_design/2 + self.lam_tgt/2
        mode_out21['z'] = 0.0
        mode_out21['x span'] = 2*self.waveguide_width
        mode_out21['z span'] = 2*self.dz_design
        
        # Design Region Monitor
        field_design = self.fdtd.addpower()
        field_design['name'] = 'design_region_monitor'
        
        if self.dimension == 2:
            field_design['monitor type'] = '2D Z-normal'
            field_design['x'] = 0.0
            field_design['y'] = 0.0
            field_design['z'] = 0.0
            field_design['x span'] = 1.1*self.dx_design
            field_design['y span'] = 1.1*self.dy_design
            
        if self.dimension == 3:
            field_design['monitor type'] = '3D'
            field_design['x'] = 0.0
            field_design['y'] = 0.0
            field_design['z'] = 0.0
            field_design['x span'] = 1.1*self.dx_design
            field_design['y span'] = 1.1*self.dy_design
            field_design['z span'] = 1.1*self.dz_design
        
        # Sources
        if self.dimension == 2:
            src_fwd = self.fdtd.addmode()
            src_fwd['name'] = 'source_fwd'
            src_fwd['direction'] = 'Forward'
            src_fwd['injection axis'] = 'y-axis'
            src_fwd['mode selection'] = 'fundamental mode'
            src_fwd['wavelength start'] = self.lam_tgt
            src_fwd['wavelength stop'] = self.lam_tgt
            
            src_fwd['x'] = 0.0
            src_fwd['y'] = -self.dy_design/2 - self.lam_tgt/2
            src_fwd['z'] = 0.0
            src_fwd['x span'] = 2*self.waveguide_width

            src_adj21 = self.fdtd.addimportedsource()
            src_adj21['name'] = 'source_adj21'
            src_adj21['direction'] = 'Backward'
            src_adj21['phase'] = 90.0
            
            src_adj21['x'] = 0.0
            src_adj21['y'] = self.dy_design/2 + self.lam_tgt/2
            src_adj21['z'] = 0.0
        
        if self.dimension == 3:
            src_fwd = self.fdtd.addmode()
            src_fwd['name'] = 'source_fwd'
            src_fwd['direction'] = 'Forward'
            src_fwd['injection axis'] = 'x-axis'
            src_fwd['mode selection'] = 'fundamental mode'
            src_fwd['wavelength start'] = self.lam_tgt
            src_fwd['wavelength stop'] = self.lam_tgt
            
            src_fwd['x'] = 0.0
            src_fwd['y'] = -self.dy_design/2 - self.lam_tgt/2
            src_fwd['z'] = 0.0
            src_fwd['x span'] = 2*self.waveguide_width
            src_fwd['z span'] = 2*self.dz_design

            src_adj21 = self.fdtd.addimportedsource()
            src_adj21['name'] = 'source_adj21'
            src_adj21['direction'] = 'Backward'
            src_adj21['phase'] = 90.0
            
            src_adj21['x'] = 0.0
            src_adj21['y'] = self.dy_design/2 + self.lam_tgt/2
            src_adj21['z'] = 0.0

        # Mesh
        mesh = self.fdtd.addmesh()
        mesh['name'] = 'mesh_design'
        
        if self.dimension == 2:
            mesh['x'] = 0.0
            mesh['y'] = 0.0
            mesh['z'] = 0.0
            mesh['x span'] = 1.1*self.dx_design
            mesh['y span'] = 1.1*self.dy_design
            mesh['dx'] = self.dx_design/10
            mesh['dy'] = self.dy_design/10
        
        elif self.dimension == 3:
            mesh['x'] = 0.0
            mesh['y'] = 0.0
            mesh['z'] = 0.0
            mesh['x span'] = 1.1*self.dx_design
            mesh['y span'] = 1.1*self.dy_design
            mesh['z span'] = 1.1*self.dz_design
            mesh['dx'] = self.dx_design/10
            mesh['dy'] = self.dy_design/10
            mesh['dz'] = self.dz_design/10

        # Waveguides
        wg_in = self.fdtd.addrect();
        wg_in['name'] = 'waveguide_in'
        wg_in['x'] = 0.0
        wg_in['x span'] = self.waveguide_width
        wg_in['y min'] = 1.1*(-self.dy_design/2 - self.lam_tgt)
        wg_in['y max'] = -self.dy_design/2
        wg_in['z'] = 0.0
        
        if self.dimension == 3:
            wg_in['z span'] = self.dz_design
        
        wg_out = self.fdtd.addrect();
        wg_out['name'] = 'waveguide_out'
        wg_out['x'] = 0.0
        wg_out['x span'] = self.waveguide_width
        wg_out['y min'] = self.dy_design/2
        wg_out['y max'] = 1.1*(self.dy_design/2 + self.lam_tgt)
        wg_out['z'] = 0.0
        
        if self.dimension == 3:
            wg_out['z span'] = self.dz_design
        
        # Generate Pattern
        x = np.linspace(-self.dx_design/2, self.dx_design/2, self.Nx)
        y = np.linspace(-self.dy_design/2, self.dy_design/2, self.Ny)
        z = np.linspace(-self.dz_design/2, self.dz_design/2, 2)
        
        self.fdtd.putv('x', x)
        self.fdtd.putv('y', y)
        self.fdtd.putv('z', z)
        self.fdtd.putv('n', np.random.rand(self.Nx, self.Ny, 2))
        
        self.fdtd.eval('addimport;')
        self.fdtd.eval('set("name", "device");')
        self.fdtd.eval('importnk2(n, x, y, z);')
        
        # Parallelization Settings
        self.fdtd.eval('setresource("FDTD", 1, "processes", "' + str(self.Nthreads) + '");')
        self.fdtd.eval('setresource("FDTD", 1, "threads", "1");')
        
        self.fdtd.eval('save;')
                
    def update_design(self, density):
        self.fdtd.switchtolayout()
        
        self.fdtd.eval('select("device");')
        self.fdtd.eval('delete;')
        
        # Update Design
        nLow = self.n_padding
        nHigh = self.n_waveguide
        dn = nHigh - nLow
        RI = np.real(nLow + dn*density)# + 1j*1e-3
        
        upsampling_ratio_fdtd = int(np.max((10*self.upsampling_ratio_fdtd, 1)))
        RI_upsampled = zoom(RI, upsampling_ratio_fdtd, order=0, mode='grid-constant')
        RI_upsampled = np.stack((RI_upsampled, RI_upsampled), axis=-1)
        
        x = np.linspace(-self.dx_design/2, self.dx_design/2, upsampling_ratio_fdtd*self.Nx)
        y = np.linspace(-self.dy_design/2, self.dy_design/2, upsampling_ratio_fdtd*self.Ny)
        z = np.linspace(-self.dz_design/2, self.dz_design/2, 2)
        
        self.fdtd.putv('x', x)
        self.fdtd.putv('y', y)
        self.fdtd.putv('z', z)
        self.fdtd.putv('n', RI_upsampled.astype(np.float64))
        
        # Generate Pattern
        self.fdtd.eval('addimport;')
        self.fdtd.eval('set("name", "device");')
        self.fdtd.eval('importnk2(n, x, y, z);')
    
    def get_cost(self, x, get_grad=False):
        #print('Get Cost', flush=True)
        x = x.reshape(self.Nx, self.Ny)
    
        # Run Forward Simulation
        #print('Update Design', flush=True)
        self.update_design(x)
        #print('Enabling Sources', flush=True)
        self.fdtd.eval('setnamed("source_fwd", "enabled", 1);')
        self.fdtd.eval('setnamed("source_adj21", "enabled", 0);')
        self.fdtd.eval('setnamed("mode_monitor21", "enabled", 0);')
        
        #print('Running Simulation', flush=True)
        while True:
            try:
                self.fdtd.run()
                x_out = np.squeeze(self.fdtd.getdata("output_monitor21", "x"))
                break
            except:
                time.sleep(60)
    
        # Output Monitor Data
        #print('Getting Data', flush=True)
        y_out = np.squeeze(self.fdtd.getdata("output_monitor21", "y"))
        z_out = np.squeeze(self.fdtd.getdata("output_monitor21", "z"))
        f_out = np.squeeze(self.fdtd.getdata("output_monitor21", "f"))
        E_out21 = np.zeros((3, x_out.size, y_out.size, z_out.size, self.N_freq)).astype(np.complex128)
        E_out21[0,:,:,:,:] = self.fdtd.getdata("output_monitor21", "Ex")
        E_out21[1,:,:,:,:] = self.fdtd.getdata("output_monitor21", "Ey")
        E_out21[2,:,:,:,:] = self.fdtd.getdata("output_monitor21", "Ez")
        H_out21 = np.zeros((3, x_out.size, y_out.size, z_out.size, self.N_freq)).astype(np.complex128)
        H_out21[0,:,:,:,:] = self.fdtd.getdata("output_monitor21", "Hx")
        H_out21[1,:,:,:,:] = self.fdtd.getdata("output_monitor21", "Hy")
        H_out21[2,:,:,:,:] = self.fdtd.getdata("output_monitor21", "Hz")
        
        x_out = np.squeeze(self.fdtd.getdata("output_monitor11", "x"))
        y_out = np.squeeze(self.fdtd.getdata("output_monitor11", "y"))
        z_out = np.squeeze(self.fdtd.getdata("output_monitor11", "z"))
        E_out11 = np.zeros((3, x_out.size, y_out.size, z_out.size, self.N_freq)).astype(np.complex128)
        E_out11[0,:,:,:,:] = self.fdtd.getdata("output_monitor11", "Ex")
        E_out11[1,:,:,:,:] = self.fdtd.getdata("output_monitor11", "Ey")
        E_out11[2,:,:,:,:] = self.fdtd.getdata("output_monitor11", "Ez")
        H_out11 = np.zeros((3, x_out.size, y_out.size, z_out.size, self.N_freq)).astype(np.complex128)
        H_out11[0,:,:,:,:] = self.fdtd.getdata("output_monitor11", "Hx")
        H_out11[1,:,:,:,:] = self.fdtd.getdata("output_monitor11", "Hy")
        H_out11[2,:,:,:,:] = self.fdtd.getdata("output_monitor11", "Hz")

        # Mode Profile Data from Source
        self.fdtd.eval('mode = getresult("source_fwd","mode profile");')
        self.fdtd.eval('Ex = mode.Ex;')
        self.fdtd.eval('Ey = mode.Ey;')
        self.fdtd.eval('Ez = mode.Ez;')
        self.fdtd.eval('Hx = mode.Hx;')
        self.fdtd.eval('Hy = mode.Hy;')
        self.fdtd.eval('Hz = mode.Hz;')
        E_mode_in = np.zeros((3, x_out.size, y_out.size, z_out.size, self.N_freq)).astype(np.complex128)
        E_mode_in[0,:,:,:,:] = self.fdtd.getv('Ex')
        E_mode_in[1,:,:,:,:] = self.fdtd.getv('Ey')
        E_mode_in[2,:,:,:,:] = self.fdtd.getv('Ez')
        H_mode_in = np.zeros((3, x_out.size, y_out.size, z_out.size, self.N_freq)).astype(np.complex128)
        H_mode_in[0,:,:,:,:] = self.fdtd.getv('Hx')
        H_mode_in[1,:,:,:,:] = self.fdtd.getv('Hy')
        H_mode_in[2,:,:,:,:] = self.fdtd.getv('Hz')
        
        #np.savez(directory + '/debug_fields', E_out21=E_out21, E_out11=E_out11, x_out=x_out, y_out=y_out, z_out=z_out, E_mode=E_mode)
        
        # Compute Overlap Integral
        #print('Computing Overlap', flush=True)
        P_mode_out = np.abs(np.sum(self.E_mode_out*np.conj(self.E_mode_out), axis=(0,1,2,3)))**2
        cost = -np.sum(np.abs(np.sum(E_out21*np.conj(self.E_mode_out), axis=(0,1,2,3)))**2/P_mode_out)
        #print(cost, flush=True)
        
#        np.savez(directory + '/debug_cost', cost=cost, P_mode=P_mode, T21=T21, T11=T11, E_out21=E_out21, E_out11=E_out11, E_mode=E_mode)
        #assert False
        
        if get_grad:
            # Design Region Field Data
            x_design = np.squeeze(self.fdtd.getdata("design_region_monitor","x"))
            y_design = np.squeeze(self.fdtd.getdata("design_region_monitor","y"))
            z_design = np.squeeze(self.fdtd.getdata("design_region_monitor","z"))
            E_fwd = np.zeros((3, x_design.size, y_design.size, z_design.size, self.N_freq)).astype(np.complex128)
            E_fwd[0,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ex")
            E_fwd[1,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ey")
            E_fwd[2,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ez")
            
            #np.savez(directory + '/debug_fields_grad', x_design=x_design, y_design=y_design, z_design=z_design, E_fwd=E_fwd)
            
            E_fwd_interp = np.zeros((3, self.Nx, self.Ny, z_design.size, self.N_freq)).astype(np.complex128)
            
            x_offset = self.dx_pix/2
            y_offset = self.dy_pix/2
            x_interp = np.linspace(-self.dx_design/2 + x_offset, self.dx_design/2 - x_offset, self.Nx)
            y_interp = np.linspace(-self.dy_design/2 + y_offset, self.dy_design/2 - y_offset, self.Ny)
            xy_interp = np.array(list(product(x_interp, y_interp)))
            
            for i in range(3):
                for j in range(z_design.size):
                    for k in range(self.N_freq):
                        fwd_Re_interp = RegularGridInterpolator((x_design, y_design), np.real(E_fwd[i,:,:,j,k]))
                        fwd_Im_interp = RegularGridInterpolator((x_design, y_design), np.imag(E_fwd[i,:,:,j,k]))
                        E_fwd_interp[i,:,:,j,k] = (fwd_Re_interp(xy_interp) + 1j*fwd_Im_interp(xy_interp)).reshape(self.Nx, self.Ny)
        
            # Run Adjoint Simulation
            self.fdtd.switchtolayout()
            self.fdtd.eval('setnamed("source_fwd", "enabled", 0);')
            
            Edot21 = np.sum(np.conj(E_out21)*self.E_mode_out, axis=(0,1,2,3))/P_mode_out
            dTdE21 = np.conj(self.E_mode_out)*Edot21[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]
            dFdT21 = np.array([-1])
            dFdE21 = dFdT21[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]*dTdE21
            
            self.fdtd.putv('x_out', x_out)
            self.fdtd.putv('y_out', self.dy_design/2 + self.lam_tgt/2)
            if self.dimension == 2:
                self.fdtd.putv('z_out', 0.0)
            elif self.dimension == 3:
                self.fdtd.putv('z_out', z_out)
            
            E_adj_interp = np.zeros((3, self.Nx, self.Ny, z_design.size, self.N_freq)).astype(np.complex128)
            for nf in range(self.N_freq):#self.N_freq
                self.fdtd.switchtolayout()
                
                self.fdtd.eval('setnamed("source_adj21", "enabled", 1);')
            
                Ex = dFdE21[0,:,:,:,nf]
                Ey = dFdE21[1,:,:,:,nf]
                Ez = dFdE21[2,:,:,:,nf]
                Hx = 0*Ex.copy()
                Hy = 0*Ey.copy()
                Hz = 0*Ez.copy()
                
                self.fdtd.putv('Ex', Ex)
                self.fdtd.putv('Ey', Ey)
                self.fdtd.putv('Ez', Ez)
                self.fdtd.putv('Hx', Hx)
                self.fdtd.putv('Hy', Hy)
                self.fdtd.putv('Hz', Hz)
                if self.N_freq == 1:
                    self.fdtd.putv('f', np.array([f_out]))
                else:
                    self.fdtd.putv('f', f_out[nf])
            
                self.fdtd.eval('EM' + str(nf) + ' = rectilineardataset("EM fields", x_out, y_out, z_out);')
                self.fdtd.eval('EM' + str(nf) + '.addparameter("lambda", c/f, "f", f);')
                self.fdtd.eval('EM' + str(nf) + '.addattribute("E", Ex, Ey, Ez);')
                self.fdtd.eval('EM' + str(nf) + '.addattribute("H", Hx, Hy, Hz);')
                self.fdtd.eval('select("source_adj21");')
                self.fdtd.eval('importdataset(EM' + str(nf) + ');')
            
                while True:
                    try:
                        self.fdtd.run()
                        x_design = np.squeeze(self.fdtd.getdata("design_region_monitor","x"))
                        break
                    except:
                        time.sleep(60)
            
                # Design Region Field Data
                y_design = np.squeeze(self.fdtd.getdata("design_region_monitor","y"))
                z_design = np.squeeze(self.fdtd.getdata("design_region_monitor","z"))
                E_adj = np.zeros((3, x_design.size, y_design.size, z_design.size)).astype(np.complex128)
                E_adj[0,:,:,:] = self.fdtd.getdata("design_region_monitor","Ex")[:,:,:,0]
                E_adj[1,:,:,:] = self.fdtd.getdata("design_region_monitor","Ey")[:,:,:,0]
                E_adj[2,:,:,:] = self.fdtd.getdata("design_region_monitor","Ez")[:,:,:,0]

                for i in range(3):
                    for j in range(z_design.size):
                        adj_Re_interp = RegularGridInterpolator((x_design, y_design), np.real(E_adj[i,:,:,j]))
                        adj_Im_interp = RegularGridInterpolator((x_design, y_design), np.imag(E_adj[i,:,:,j]))
                        E_adj_interp[i,:,:,j,nf] = (adj_Re_interp(xy_interp) + 1j*adj_Im_interp(xy_interp)).reshape(self.Nx, self.Ny)
                
                #np.savez(directory + '/debug_source_adj', dFdE21=dFdE21, dFdE11=dFdE11, dTdE21=dTdE21, dTdE11=dTdE11, E_mode=E_mode, E_adj_interp=E_adj_interp, E_fwd_interp=E_fwd_interp)
#                if nf == 2:
#                    assert False
        
            nLow = self.n_padding
            nHigh = self.n_waveguide
            dn = nHigh - nLow
            RI = np.real(nLow + dn*x)
            
            dedr = 2*RI*dn
            dE_xy = np.sum(E_fwd_interp*E_adj_interp, axis=(0,3,4))
            jac = 2*np.real(dE_xy*dedr)
            
            return cost, jac
        
        else:
            return cost

    def get_power_and_fields(self, x):
        x = x.reshape(self.Nx, self.Ny)
    
        # Run Forward Simulation
        self.update_design(x)
        self.fdtd.eval('setnamed("source_fwd", "enabled", 1);')
        self.fdtd.eval('setnamed("source_adj21", "enabled", 0);')
        self.fdtd.eval('setnamed("mode_monitor21", "enabled", 1);')
        
        self.fdtd.eval('select("design_region_monitor");')
        self.fdtd.eval('set("x min", ' + str(-self.dx_design/2 - self.lam_tgt/2) + ');')
        self.fdtd.eval('set("x max", ' + str(self.dx_design/2 + self.lam_tgt/2) + ');')
        self.fdtd.eval('set("y min", ' + str(-self.dy_design/2 - self.lam_tgt) + ');')
        self.fdtd.eval('set("y max", ' + str(self.dy_design/2 + self.lam_tgt) + ');')
        
        while True:
            try:
                self.fdtd.run()
                self.fdtd.eval('res = getresult("mode_monitor21", "expansion for input");')
                break
            except:
                time.sleep(60)
    
        # Mode Monitor Data
        self.fdtd.eval('T21 = res.T_forward;')
        T21 = self.fdtd.getv('T21')
        loss = 10*np.log10(1/T21)
        
        # Design Region Field Data
        x_design = np.squeeze(self.fdtd.getdata("design_region_monitor","x"))
        y_design = np.squeeze(self.fdtd.getdata("design_region_monitor","y"))
        z_design = np.squeeze(self.fdtd.getdata("design_region_monitor","z"))
        E_fwd = np.zeros((3, x_design.size, y_design.size, z_design.size, self.N_freq)).astype(np.complex128)
        E_fwd[0,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ex")
        E_fwd[1,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ey")
        E_fwd[2,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ez")
        
        return T21, loss, E_fwd, x_design, y_design