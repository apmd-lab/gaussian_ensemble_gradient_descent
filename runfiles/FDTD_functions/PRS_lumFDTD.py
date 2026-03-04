import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles')

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
                 IPR_exponent,
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
        self.symmetric_bounds = symmetric_bounds

        # Materials
        self.mat_padding = mat_padding
        self.mat_waveguide = mat_waveguide

        # Other Simulation Settings
        self.dimension = dimension
        self.Nthreads = Nthreads
        self.fsp_suffix = fsp_suffix
        self.IPR_exponent = IPR_exponent

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
        self.fdtd.eval('select("waveguide_outTE0");')
        self.fdtd.eval('set("index", ' + str(self.n_waveguide) + ');')
        self.fdtd.eval('select("waveguide_outTM0");')
        self.fdtd.eval('set("index", ' + str(self.n_waveguide) + ');')
        
        # Get Target Mode Profile
        self.fdtd.eval('select("source_fwdTE0");')
        self.fdtd.eval('set("mode selection", "user select");')
        self.fdtd.eval('set("selected mode number", 1);') # TE0 mode
        self.fdtd.eval('mode = getresult("source_fwdTE0","mode profile");')
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
        self.fdtd.eval('save("%s");' %(directory[:-15] + '/PRS_' + self.fsp_suffix + '.fsp'))
            
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
        '''
        movie = self.fdtd.addmovie()
        movie['monitor type'] = 3
        movie['x'] = 0.0
        movie['y'] = 0.0
        movie['z'] = 0.0
        movie['x span'] = self.dx_design + self.lam_tgt
        movie['y span'] = self.dy_design + 2*self.lam_tgt
        movie['lock aspect ratio'] = 1
        '''
        
        # Output Monitor
        field_outTE0 = self.fdtd.addpower()
        field_outTE0['name'] = 'output_monitorTE0'
        
        if self.dimension == 2:
            field_outTE0['monitor type'] = 'Linear X'
            field_outTE0['x min'] = 0.0
            field_outTE0['x max'] = self.dx_design/2
            field_outTE0['y'] = self.dy_design/2 + self.lam_tgt/2
            field_outTE0['z'] = 0.0
        
        if self.dimension == 3:
            field_outTE0['monitor type'] = '2D Y-normal'
            field_outTE0['x min'] = 0.0
            field_outTE0['x max'] = self.dx_design/2
            field_outTE0['y'] = self.dy_design/2 + self.lam_tgt/2
            field_outTE0['z'] = 0.0
            field_outTE0['z span'] = 2*self.dz_design
        
        field_outTM0 = self.fdtd.addpower()
        field_outTM0['name'] = 'output_monitorTM0'
        
        if self.dimension == 2:
            field_outTM0['monitor type'] = 'Linear X'
            field_outTM0['x min'] = -self.dx_design/2
            field_outTM0['x max'] = 0.0
            field_outTM0['y'] = self.dy_design/2 + self.lam_tgt/2
            field_outTM0['z'] = 0.0
        
        if self.dimension == 3:
            field_outTM0['monitor type'] = '2D Y-normal'
            field_outTM0['x min'] = -self.dx_design/2
            field_outTM0['x max'] = 0.0
            field_outTM0['y'] = -self.dy_design/2 - 0.6*self.lam_tgt
            field_outTM0['z'] = 0.0
            field_outTM0['z span'] = 2*self.dz_design
        
        mode_outTE0 = self.fdtd.addmodeexpansion()
        mode_outTE0['name'] = 'mode_monitorTE0'
        mode_outTE0['mode selection'] = 'user select'
        self.fdtd.eval('set("override global monitor settings", 0);')
        self.fdtd.eval('set("selected mode numbers", 1);')
        self.fdtd.eval('setexpansion("input", "output_monitorTE0");')

        mode_outTE0['monitor type'] = '2D Y-normal'
        mode_outTE0['x min'] = 0.0
        mode_outTE0['x max'] = self.dx_design/2
        mode_outTE0['y'] = self.dy_design/2 + self.lam_tgt/2
        mode_outTE0['z'] = 0.0
        mode_outTE0['z span'] = 2*self.dz_design

        mode_outTM0 = self.fdtd.addmodeexpansion()
        mode_outTM0['name'] = 'mode_monitorTM0'
        mode_outTM0['mode selection'] = 'user select'
        self.fdtd.eval('set("override global monitor settings", 0);')
        self.fdtd.eval('set("selected mode numbers", 2);')
        self.fdtd.eval('setexpansion("input", "output_monitorTM0");')

        mode_outTM0['monitor type'] = '2D Y-normal'
        mode_outTM0['x min'] = -self.dx_design/2
        mode_outTM0['x max'] = 0.0
        mode_outTM0['y'] = self.dy_design/2 + self.lam_tgt/2
        mode_outTM0['z'] = 0.0
        mode_outTM0['z span'] = 2*self.dz_design
        
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
            src_fwdTE0 = self.fdtd.addmode()
            src_fwdTE0['name'] = 'source_fwdTE0'
            src_fwdTE0['direction'] = 'Forward'
            src_fwdTE0['injection axis'] = 'y-axis'
            src_fwdTE0['wavelength start'] = self.lam_tgt
            src_fwdTE0['wavelength stop'] = self.lam_tgt
            
            src_fwdTE0['x min'] = 0.0
            src_fwdTE0['x max'] = self.dx_design/2
            src_fwdTE0['y'] = -self.dy_design/2 - self.lam_tgt/2
            src_fwdTE0['z'] = 0.0

            self.fdtd.eval('set("mode selection", "user select");')
            self.fdtd.eval('set("selected mode number", 1);')

            src_fwdTM0 = self.fdtd.addmode()
            src_fwdTM0['name'] = 'source_fwdTM0'
            src_fwdTM0['direction'] = 'Forward'
            src_fwdTM0['injection axis'] = 'y-axis'
            src_fwdTM0['wavelength start'] = self.lam_tgt
            src_fwdTM0['wavelength stop'] = self.lam_tgt
            
            src_fwdTM0['x min'] = 0.0
            src_fwdTM0['x max'] = self.dx_design/2
            src_fwdTM0['y'] = -self.dy_design/2 - self.lam_tgt/2
            src_fwdTM0['z'] = 0.0

            self.fdtd.eval('set("mode selection", "user select");')
            self.fdtd.eval('set("selected mode number", 2);')

            src_adjTE0 = self.fdtd.addimportedsource()
            src_adjTE0['name'] = 'source_adjTE0'
            src_adjTE0['direction'] = 'Backward'
            src_adjTE0['phase'] = 90.0
            
            src_adjTE0['x'] = self.dx_design*3/4
            src_adjTE0['y'] = self.dy_design/2 + self.lam_tgt/2
            src_adjTE0['z'] = 0.0

            src_adjTM0 = self.fdtd.addimportedsource()
            src_adjTM0['name'] = 'source_adjTM0'
            src_adjTM0['direction'] = 'Backward'
            src_adjTM0['phase'] = 90.0
            
            src_adjTM0['x'] = self.dx_design/4
            src_adjTM0['y'] = self.dy_design/2 + self.lam_tgt/2
            src_adjTM0['z'] = 0.0
        
        if self.dimension == 3:
            src_fwdTE0 = self.fdtd.addmode()
            src_fwdTE0['name'] = 'source_fwdTE0'
            src_fwdTE0['direction'] = 'Forward'
            src_fwdTE0['injection axis'] = 'y-axis'
            src_fwdTE0['mode selection'] = 'fundamental mode'
            src_fwdTE0['wavelength start'] = self.lam_tgt
            src_fwdTE0['wavelength stop'] = self.lam_tgt
            
            src_fwdTE0['x min'] = 0.0
            src_fwdTE0['x max'] = self.dx_design/2
            src_fwdTE0['y'] = -self.dy_design/2 - self.lam_tgt/2
            src_fwdTE0['z'] = 0.0
            src_fwdTE0['z span'] = 2*self.dz_design

            src_fwdTM0 = self.fdtd.addmode()
            src_fwdTM0['name'] = 'source_fwdTM0'
            src_fwdTM0['direction'] = 'Forward'
            src_fwdTM0['injection axis'] = 'y-axis'
            src_fwdTM0['mode selection'] = 'fundamental mode'
            src_fwdTM0['wavelength start'] = self.lam_tgt
            src_fwdTM0['wavelength stop'] = self.lam_tgt
            
            src_fwdTM0['x min'] = 0.0
            src_fwdTM0['x max'] = self.dx_design/2
            src_fwdTM0['y'] = -self.dy_design/2 - self.lam_tgt/2
            src_fwdTM0['z'] = 0.0
            src_fwdTM0['z span'] = 2*self.dz_design

            src_adjTE0 = self.fdtd.addimportedsource()
            src_adjTE0['name'] = 'source_adjTE0'
            src_adjTE0['direction'] = 'Backward'
            src_adjTE0['phase'] = 90.0
            
            src_adjTE0['x'] = 0.0
            src_adjTE0['y'] = self.dy_design/2 + self.lam_tgt/2
            src_adjTE0['z'] = 0.0

            src_adjTM0 = self.fdtd.addimportedsource()
            src_adjTM0['name'] = 'source_adjTM0'
            src_adjTM0['direction'] = 'Backward'
            src_adjTM0['phase'] = 90.0
            
            src_adjTM0['x'] = 0.0
            src_adjTM0['y'] = self.dy_design/2 + self.lam_tgt/2
            src_adjTM0['z'] = 0.0

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
        wg_in = self.fdtd.addrect()
        wg_in['name'] = 'waveguide_in'
        wg_in['x min'] = self.dx_design/10
        wg_in['x max'] = self.dx_design/10 + self.waveguide_width
        wg_in['y min'] = 1.1*(-self.dy_design/2 - self.lam_tgt)
        wg_in['y max'] = -self.dy_design/2
        wg_in['z'] = 0.0
        
        if self.dimension == 3:
            wg_in['z span'] = self.dz_design
        
        wg_outTE0 = self.fdtd.addrect()
        wg_outTE0['name'] = 'waveguide_outTE0'
        wg_outTE0['x min'] = self.dx_design/10
        wg_outTE0['x max'] = self.dx_design/10 + self.waveguide_width
        wg_outTE0['y min'] = self.dy_design/2
        wg_outTE0['y max'] = 1.1*(self.dy_design/2 + self.lam_tgt)
        wg_outTE0['z'] = 0.0
        
        if self.dimension == 3:
            wg_outTE0['z span'] = self.dz_design
        
        wg_outTM0 = self.fdtd.addrect()
        wg_outTM0['name'] = 'waveguide_outTM0'
        wg_outTM0['x min'] = -self.dx_design/10 - self.waveguide_width
        wg_outTM0['x max'] = -self.dx_design/10
        wg_outTM0['y min'] = self.dy_design/2
        wg_outTM0['y max'] = 1.1*(self.dy_design/2 + self.lam_tgt)
        wg_outTM0['z'] = 0.0
        
        if self.dimension == 3:
            wg_outTM0['z span'] = self.dz_design
        
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
        RI_upsampled = zoom(RI, upsampling_ratio_fdtd, order=1, mode='grid-constant')
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
        
        #print('Running TE0 Simulation', flush=True)
        self.fdtd.eval('setnamed("source_fwdTE0", "enabled", 1);')
        self.fdtd.eval('setnamed("source_fwdTM0", "enabled", 0);')
        self.fdtd.eval('setnamed("source_adjTE0", "enabled", 0);')
        self.fdtd.eval('setnamed("source_adjTM0", "enabled", 0);')
        self.fdtd.eval('setnamed("mode_monitorTE0", "enabled", 1);')
        self.fdtd.eval('setnamed("mode_monitorTM0", "enabled", 0);')
        
        while True:
            try:
                self.fdtd.run()
                x_outTE0 = np.squeeze(self.fdtd.getdata("output_monitorTE0", "x"))
                break
            except:
                time.sleep(60)
    
        # Output Monitor Data (TE0)
        #print('Getting Data (TE0)', flush=True)
        y_outTE0 = np.squeeze(self.fdtd.getdata("output_monitorTE0", "y"))
        z_outTE0 = np.squeeze(self.fdtd.getdata("output_monitorTE0", "z"))
        f_out = np.squeeze(self.fdtd.getdata("output_monitorTE0", "f"))
        E_outTE0 = np.zeros((3, x_outTE0.size, y_outTE0.size, z_outTE0.size, self.N_freq)).astype(np.complex128)
        E_outTE0[0,:,:,:,:] = self.fdtd.getdata("output_monitorTE0", "Ex")
        E_outTE0[1,:,:,:,:] = self.fdtd.getdata("output_monitorTE0", "Ey")
        E_outTE0[2,:,:,:,:] = self.fdtd.getdata("output_monitorTE0", "Ez")
        H_outTE0 = np.zeros((3, x_outTE0.size, y_outTE0.size, z_outTE0.size, self.N_freq)).astype(np.complex128)
        H_outTE0[0,:,:,:,:] = self.fdtd.getdata("output_monitorTE0", "Hx")
        H_outTE0[1,:,:,:,:] = self.fdtd.getdata("output_monitorTE0", "Hy")
        H_outTE0[2,:,:,:,:] = self.fdtd.getdata("output_monitorTE0", "Hz")

        if get_grad:
            # Design Region Field Data
            x_design = np.squeeze(self.fdtd.getdata("design_region_monitor","x"))
            y_design = np.squeeze(self.fdtd.getdata("design_region_monitor","y"))
            z_design = np.squeeze(self.fdtd.getdata("design_region_monitor","z"))
            E_fwdTE0 = np.zeros((3, x_design.size, y_design.size, z_design.size, self.N_freq)).astype(np.complex128)
            E_fwdTE0[0,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ex")
            E_fwdTE0[1,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ey")
            E_fwdTE0[2,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ez")

        self.fdtd.switchtolayout()

        #print('Running TM0 Simulation', flush=True)
        self.fdtd.eval('setnamed("source_fwdTE0", "enabled", 0);')
        self.fdtd.eval('setnamed("source_fwdTM0", "enabled", 1);')
        self.fdtd.eval('setnamed("source_adjTE0", "enabled", 0);')
        self.fdtd.eval('setnamed("source_adjTM0", "enabled", 0);')
        self.fdtd.eval('setnamed("mode_monitorTE0", "enabled", 0);')
        self.fdtd.eval('setnamed("mode_monitorTM0", "enabled", 1);')
        
        while True:
            try:
                self.fdtd.run()
                x_outTM0 = np.squeeze(self.fdtd.getdata("output_monitorTM0", "x"))
                break
            except:
                time.sleep(60)
        
        # Output Monitor Data (TM0)
        #print('Getting Data (TM0)', flush=True)
        y_outTM0 = np.squeeze(self.fdtd.getdata("output_monitorTM0", "y"))
        z_outTM0 = np.squeeze(self.fdtd.getdata("output_monitorTM0", "z"))
        E_outTM0 = np.zeros((3, x_outTM0.size, y_outTM0.size, z_outTM0.size, self.N_freq)).astype(np.complex128)
        E_outTM0[0,:,:,:,:] = self.fdtd.getdata("output_monitorTM0", "Ex")
        E_outTM0[1,:,:,:,:] = self.fdtd.getdata("output_monitorTM0", "Ey")
        E_outTM0[2,:,:,:,:] = self.fdtd.getdata("output_monitorTM0", "Ez")
        H_outTM0 = np.zeros((3, x_outTM0.size, y_outTM0.size, z_outTM0.size, self.N_freq)).astype(np.complex128)
        H_outTM0[0,:,:,:,:] = self.fdtd.getdata("output_monitorTM0", "Hx")
        H_outTM0[1,:,:,:,:] = self.fdtd.getdata("output_monitorTM0", "Hy")
        H_outTM0[2,:,:,:,:] = self.fdtd.getdata("output_monitorTM0", "Hz")

        if get_grad:
            # Design Region Field Data
            x_design = np.squeeze(self.fdtd.getdata("design_region_monitor","x"))
            y_design = np.squeeze(self.fdtd.getdata("design_region_monitor","y"))
            z_design = np.squeeze(self.fdtd.getdata("design_region_monitor","z"))
            E_fwdTM0 = np.zeros((3, x_design.size, y_design.size, z_design.size, self.N_freq)).astype(np.complex128)
            E_fwdTM0[0,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ex")
            E_fwdTM0[1,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ey")
            E_fwdTM0[2,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ez")
        
        #np.savez(directory + '/debug_fields', E_out21=E_out21, E_out11=E_out11, x_out=x_out, y_out=y_out, z_out=z_out)
        
        # Compute Overlap Integral
        #print('Computing Overlap', flush=True)
        if np.abs(self.E_mode_out.shape[1] - E_outTE0.shape[1]) == 0:
            E_mode_outTE0 = self.E_mode_out.copy()
        elif np.abs(self.E_mode_out.shape[1] - E_outTE0.shape[1]) == 1:
            E_mode_outTE0 = self.E_mode_out[:,:-1,:,:,:]
        elif np.abs(self.E_mode_out.shape[1] - E_outTE0.shape[1]) == 2:
            E_mode_outTE0 = self.E_mode_out[:,1:-1,:,:,:]
        
        if np.abs(self.E_mode_out.shape[1] - E_outTM0.shape[1]) == 0:
            E_mode_outTM0 = self.E_mode_out.copy()
        elif np.abs(self.E_mode_out.shape[1] - E_outTM0.shape[1]) == 1:
            E_mode_outTM0 = self.E_mode_out[:,:-1,:,:,:]
        elif np.abs(self.E_mode_out.shape[1] - E_outTM0.shape[1]) == 2:
            E_mode_outTM0 = self.E_mode_out[:,1:-1,:,:,:]

        P_mode_out = np.abs(np.sum(E_mode_outTE0*np.conj(E_mode_outTE0), axis=(0,1,2,3)))**2
        T_outTE0 = np.sum(np.abs(np.sum(E_outTE0*np.conj(E_mode_outTE0), axis=(0,1,2,3)))**2/P_mode_out)
        T_outTM0 = np.sum(np.abs(np.sum(E_outTM0*np.conj(E_mode_outTM0), axis=(0,1,2,3)))**2/P_mode_out)
        cost = -np.mean(T_outTE0**self.IPR_exponent + T_outTM0**self.IPR_exponent)
        #print(cost, flush=True)
        
#        np.savez(directory + '/debug_cost', cost=cost, P_mode=P_mode, T21=T21, T11=T11, E_out21=E_out21, E_out11=E_out11)
        
        if get_grad:
            # Design Region Field Data
            E_fwd = E_fwdTE0 + E_fwdTM0
            
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
            self.fdtd.eval('setnamed("source_fwdTE0", "enabled", 0);')
            self.fdtd.eval('setnamed("source_fwdTM0", "enabled", 0);')
            
            EdotTE0 = np.sum(np.conj(E_outTE0)*E_mode_outTE0, axis=(0,1,2,3))/P_mode_out
            EdotTM0 = np.sum(np.conj(E_outTM0)*E_mode_outTM0, axis=(0,1,2,3))/P_mode_out
            dTdETE0 = np.conj(E_mode_outTE0)*EdotTE0[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]
            dTdETM0 = np.conj(E_mode_outTM0)*EdotTM0[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]
            dFdTTE0 = np.array([-self.IPR_exponent / 2 * T_outTE0**(self.IPR_exponent - 1)])
            dFdTTM0 = np.array([-self.IPR_exponent / 2 * T_outTM0**(self.IPR_exponent - 1)])
            dFdETE0 = dFdTTE0[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]*dTdETE0
            dFdETM0 = dFdTTM0[np.newaxis,np.newaxis,np.newaxis,np.newaxis,:]*dTdETM0

            self.fdtd.putv('x_outTE0', x_outTE0)
            self.fdtd.putv('y_outTE0', self.dy_design/2 + self.lam_tgt/2)
            if self.dimension == 2:
                self.fdtd.putv('z_outTE0', 0.0)
            elif self.dimension == 3:
                self.fdtd.putv('z_outTE0', z_outTE0)
            
            self.fdtd.putv('x_outTM0', x_outTM0)
            self.fdtd.putv('y_outTM0', self.dy_design/2 + self.lam_tgt/2)
            if self.dimension == 2:
                self.fdtd.putv('z_outTM0', 0.0)
            elif self.dimension == 3:
                self.fdtd.putv('z_outTM0', z_outTM0)
            
            E_adj_interp = np.zeros((3, self.Nx, self.Ny, z_design.size, self.N_freq)).astype(np.complex128)
            for nf in range(self.N_freq):#self.N_freq
                self.fdtd.switchtolayout()
                
                self.fdtd.eval('setnamed("source_adjTE0", "enabled", 1);')
                self.fdtd.eval('setnamed("source_adjTM0", "enabled", 1);')
            
                ExTE0 = dFdETE0[0,:,:,:,nf]
                EyTE0 = dFdETE0[1,:,:,:,nf]
                EzTE0 = dFdETE0[2,:,:,:,nf]
                HxTE0 = 0*ExTE0.copy()
                HyTE0 = 0*EyTE0.copy()
                HzTE0 = 0*EzTE0.copy()

                ExTM0 = dFdETM0[0,:,:,:,nf]
                EyTM0 = dFdETM0[1,:,:,:,nf]
                EzTM0 = dFdETM0[2,:,:,:,nf]
                HxTM0 = 0*ExTM0.copy()
                HyTM0 = 0*EyTM0.copy()
                HzTM0 = 0*EzTM0.copy()
                
                self.fdtd.putv('ExTE0', ExTE0)
                self.fdtd.putv('EyTE0', EyTE0)
                self.fdtd.putv('EzTE0', EzTE0)
                self.fdtd.putv('HxTE0', HxTE0)
                self.fdtd.putv('HyTE0', HyTE0)
                self.fdtd.putv('HzTE0', HzTE0)
                if self.N_freq == 1:
                    self.fdtd.putv('f', np.array([f_out]))
                else:
                    self.fdtd.putv('f', f_out[nf])
                
                self.fdtd.putv('ExTM0', ExTM0)
                self.fdtd.putv('EyTM0', EyTM0)
                self.fdtd.putv('EzTM0', EzTM0)
                self.fdtd.putv('HxTM0', HxTM0)
                self.fdtd.putv('HyTM0', HyTM0)
                self.fdtd.putv('HzTM0', HzTM0)
                if self.N_freq == 1:
                    self.fdtd.putv('f', np.array([f_out]))
                else:
                    self.fdtd.putv('f', f_out[nf])
            
                self.fdtd.eval('EM' + str(nf) + ' = rectilineardataset("EM fields", x_outTE0, y_outTE0, z_outTE0);')
                self.fdtd.eval('EM' + str(nf) + '.addparameter("lambda", c/f, "f", f);')
                self.fdtd.eval('EM' + str(nf) + '.addattribute("E", ExTE0, EyTE0, EzTE0);')
                self.fdtd.eval('EM' + str(nf) + '.addattribute("H", HxTE0, HyTE0, HzTE0);')
                self.fdtd.eval('select("source_adjTE0");')
                self.fdtd.eval('importdataset(EM' + str(nf) + ');')
                self.fdtd.eval('set("x", ' + str(-self.dx_design/5) + ');')

                self.fdtd.eval('EM' + str(nf) + ' = rectilineardataset("EM fields", x_outTM0, y_outTM0, z_outTM0);')
                self.fdtd.eval('EM' + str(nf) + '.addparameter("lambda", c/f, "f", f);')
                self.fdtd.eval('EM' + str(nf) + '.addattribute("E", ExTM0, EyTM0, EzTM0);')
                self.fdtd.eval('EM' + str(nf) + '.addattribute("H", HxTM0, HyTM0, HzTM0);')
                self.fdtd.eval('select("source_adjTM0");')
                self.fdtd.eval('importdataset(EM' + str(nf) + ');')
                self.fdtd.eval('set("x", ' + str(self.dx_design/5) + ');')
            
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
        self.update_design(x)

        self.fdtd.eval('select("design_region_monitor");')
        self.fdtd.eval('set("x min", ' + str(-self.dx_design/2 - self.lam_tgt/2) + ');')
        self.fdtd.eval('set("x max", ' + str(self.dx_design/2 + self.lam_tgt/2) + ');')
        self.fdtd.eval('set("y min", ' + str(-self.dy_design/2 - self.lam_tgt) + ');')
        self.fdtd.eval('set("y max", ' + str(self.dy_design/2 + self.lam_tgt) + ');')

        # Run TE0 Simulation
        self.fdtd.eval('setnamed("source_fwdTE0", "enabled", 1);')
        self.fdtd.eval('setnamed("source_fwdTM0", "enabled", 0);')
        self.fdtd.eval('setnamed("source_adjTE0", "enabled", 0);')
        self.fdtd.eval('setnamed("source_adjTM0", "enabled", 0);')
        self.fdtd.eval('setnamed("mode_monitorTE0", "enabled", 1);')
        self.fdtd.eval('setnamed("mode_monitorTM0", "enabled", 0);')
        
        while True:
            try:
                self.fdtd.run()
                self.fdtd.eval('res = getresult("mode_monitorTE0", "expansion for input");')
                break
            except:
                time.sleep(60)
    
        # Mode Monitor Data
        self.fdtd.eval('T_outTE0 = res.T_forward;')
        T_outTE0 = self.fdtd.getv('T_outTE0')
        lossTE0 = 10*np.log10(1/T_outTE0)
        
        # Design Region Field Data
        x_design = np.squeeze(self.fdtd.getdata("design_region_monitor","x"))
        y_design = np.squeeze(self.fdtd.getdata("design_region_monitor","y"))
        z_design = np.squeeze(self.fdtd.getdata("design_region_monitor","z"))
        E_fwdTE0 = np.zeros((3, x_design.size, y_design.size, z_design.size, self.N_freq)).astype(np.complex128)
        E_fwdTE0[0,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ex")
        E_fwdTE0[1,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ey")
        E_fwdTE0[2,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ez")

        # Run TM0 Simulation
        self.fdtd.eval('setnamed("source_fwdTE0", "enabled", 0);')
        self.fdtd.eval('setnamed("source_fwdTM0", "enabled", 1);')
        self.fdtd.eval('setnamed("source_adjTE0", "enabled", 0);')
        self.fdtd.eval('setnamed("source_adjTM0", "enabled", 0);')
        self.fdtd.eval('setnamed("mode_monitorTE0", "enabled", 0);')
        self.fdtd.eval('setnamed("mode_monitorTM0", "enabled", 1);')
        
        while True:
            try:
                self.fdtd.run()
                self.fdtd.eval('res = getresult("mode_monitorTM0", "expansion for input");')
                break
            except:
                time.sleep(60)
    
        # Mode Monitor Data
        self.fdtd.eval('T_outTM0 = res.T_forward;')
        T_outTM0 = self.fdtd.getv('T_outTM0')
        lossTM0 = 10*np.log10(1/T_outTM0)
        
        # Design Region Field Data
        x_design = np.squeeze(self.fdtd.getdata("design_region_monitor","x"))
        y_design = np.squeeze(self.fdtd.getdata("design_region_monitor","y"))
        z_design = np.squeeze(self.fdtd.getdata("design_region_monitor","z"))
        E_fwdTM0 = np.zeros((3, x_design.size, y_design.size, z_design.size, self.N_freq)).astype(np.complex128)
        E_fwdTM0[0,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ex")
        E_fwdTM0[1,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ey")
        E_fwdTM0[2,:,:,:,:] = self.fdtd.getdata("design_region_monitor","Ez")
        
        return T_outTE0, T_outTM0, lossTE0, lossTM0, E_fwdTE0, E_fwdTM0, x_design, y_design