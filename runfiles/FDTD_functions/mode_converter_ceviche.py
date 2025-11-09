import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-15])

import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import time
import subprocess
import util.read_mat_data as rmd

import autograd.numpy as npa
import ceviche
from ceviche import fdfd_ez, jacobian
from ceviche.modes import insert_mode
import collections
Slice = collections.namedtuple('Slice', 'x y')

#import warnings
#warnings.filterwarnings('ignore')

from mpi4py import MPI
comm = MPI.COMM_WORLD

c = 299792458
um = 1e-6
fs = 1e-15

class custom_objective:
    def __init__(self,
                 lam_tgt, # in um
                 design_dim, # in um
                 Nx,
                 Ny,
                 waveguide_width, # in um
                 mat_padding,
                 mat_waveguide,
                 total_sim_time=1e3 # in fs
                 ):
    
        # Source
        self.lam_tgt = lam_tgt*um
        
        # Geometries
        self.dx_design, self.dy_design = design_dim*um
        self.dx_pix = self.dx_design/Nx
        self.Nx = Nx
        self.Ny = Ny
        self.waveguide_width = waveguide_width*um
        
        # Load Material Refractive Indices
        mat_type = np.array([mat_padding, mat_waveguide])
        raw_wavelength, mat_dict = rmd.load_all(np.array([self.lam_tgt*1e9]), 'n_k', mat_type)
        self.n_padding = np.real(mat_dict[mat_padding])[0]
        self.n_waveguide = np.real(mat_dict[mat_waveguide])[0]

        # Other Simulation Settings
        self.pml_size = self.lam_tgt*um
        self.total_sim_time = total_sim_time*fs
        
    def set_accuracy(self, upsampling_ratio):
        self.upsampling_ratio = upsampling_ratio
        self.resolution = self.upsampling_ratio/self.dx_pix
        
        ### Dimensions
        self.Npml = int(self.pml_size*self.resolution)
        self.Nlam_half = int(self.lam_tgt/2*self.resolution)
        self.Nx_up = int(self.Nx*self.upsampling_ratio)
        self.Ny_up = int(self.Ny*self.upsampling_ratio)
        
        Sx = 2*self.Npml + 4*self.Nlam_half + self.Nx_up
        Sy = 2*self.Npml + 2*self.Nlam_half + self.Ny_up
        
        center_y = int(np.floor(self.Ny_up/2 + 0.5)) + self.Npml + self.Nlam_half
        
        waveguide_width_npix = np.round(self.waveguide_width*self.resolution)
        if self.Nx % 2 == 0:
            waveguide_width_npix += (waveguide_width_npix % 2)*np.sign(self.waveguide_width*self.resolution - waveguide_width_npix)
        else:
            waveguide_width_npix += (waveguide_width_npix % 2 == 0)*np.sign(self.waveguide_width*self.resolution - waveguide_width_npix)
        waveguide_halfwidth_npix = int(np.floor(waveguide_width_npix/2 + 0.5))
        waveguide_bottom = int(center_y - waveguide_halfwidth_npix)
        waveguide_top = int(waveguide_bottom + waveguide_width_npix)
                                         
        ### Geometries
        self.density_background = np.zeros((Sx, Sy))
        self.mask_design = np.zeros((Sx, Sy))
        density_design = np.zeros((Sx, Sy))
        
        # Input Waveguide
        self.density_background[
            :self.Npml+2*self.Nlam_half,
            waveguide_bottom:waveguide_top,
        ] = 1
        
        # Output Waveguide
        self.density_background[
            -self.Npml-2*self.Nlam_half:,
            waveguide_bottom:waveguide_top,
        ] = 1
        
        # Design Region
        self.mask_design[
            self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
            self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
        ] = 1
        density_design[
            self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
            self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
        ] = 1
        
        # Permittivity Map
        density_combined = density_design*self.mask_design + self.density_background*(self.mask_design==0).astype(npa.float64)
        eps = (self.n_waveguide**2 - self.n_padding**2)*density_combined + self.n_padding**2
        
        # Source
        omega = 2*np.pi*c/self.lam_tgt
        
        self.source_plane = Slice(
            x = np.array([int(self.Npml + self.Nlam_half)]),
            y = np.arange(center_y - waveguide_width_npix, center_y + waveguide_width_npix).astype(np.int32),
        )
        
        self.source = insert_mode(
            omega,
            1/self.resolution,
            self.source_plane.x,
            self.source_plane.y,
            eps,
            m=1,
        )
        
        # Monitor
        self.monitor_plane = Slice(
            x = np.array([int(Sx - self.Npml - self.Nlam_half)]),
            y = np.arange(center_y - waveguide_width_npix, center_y + waveguide_width_npix).astype(np.int32),
        )
        
        self.monitor = insert_mode(
            omega,
            1/self.resolution,
            self.monitor_plane.x,
            self.monitor_plane.y,
            eps,
            m=2,
        )
        
        # Simulation
        self.simulation = fdfd_ez(omega, 1/self.resolution, eps, [self.Npml, self.Npml])
    
    def visualize(self, Ez, eps):
        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6,3))
        ceviche.viz.real(Ez, outline=eps, ax=ax[0], cbar=False)
        ax[0].plot(self.source_plane.x*np.ones(len(self.source_plane.y)), self.source_plane.y, 'r-')
        ax[0].plot(self.monitor_plane.x*np.ones(len(self.monitor_plane.y)), self.monitor_plane.y, 'b-')
        ceviche.viz.abs(eps, ax=ax[1], cmap='Greys')
        plt.savefig(directory + '/simulation_visualization')
        plt.close()
    
    def cost(self, density_design):
        density_combined = density_design*self.mask_design + self.density_background*(self.mask_design==0).astype(npa.float64)
        eps = (self.n_waveguide**2 - self.n_padding**2)*density_combined + self.n_padding**2
        
        self.simulation.eps_r = eps
        _,_, Ez = self.simulation.solve(self.source)
    
        numer = npa.abs(npa.sum(npa.conj(Ez)*self.monitor))
        denom = npa.sqrt(npa.abs(npa.sum(npa.conj(Ez)*Ez))\
                        *npa.abs(npa.sum(npa.conj(self.monitor)*self.monitor)))
        
        cost = numer/denom
        
#        print(npa.abs(npa.sum(npa.conj(Ez)*self.monitor)), flush=True)
#        print(npa.abs(npa.sum(npa.conj(Ez)*Ez)), flush=True)
#        print(npa.abs(npa.sum(npa.conj(self.monitor)*self.monitor)), flush=True)
        
        # Plot Simulation (debugging)
#        self.visualize(Ez, eps)
#        assert False
        
        return cost
    
    def get_cost(self, x, get_grad=False):
        x_upsampled = zoom(x.reshape(self.Nx, self.Ny).astype(np.float64), self.upsampling_ratio, order=0)
        
        density_design = npa.zeros_like(self.density_background).astype(npa.float64)
        density_design[
            self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
            self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
        ] = x_upsampled
        
        cost = self.cost(density_design)
        
        if get_grad:
            jac = jacobian(self.cost, mode='reverse')(density_design)
            jac = jac[0,:].reshape(density_design.shape)
            jac = jac[
                self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
                self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
            ]
#            np.savez(directory + '/debug_jacobian', jac=jac)
#            assert False
            
            return cost, jac
        else:
            return cost