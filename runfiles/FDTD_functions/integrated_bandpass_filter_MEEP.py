import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, directory[:-15])

import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import time
import subprocess
import util.read_mat_data as rmd
from fractions import Fraction

import meep as mp
import meep.adjoint as mpa
import autograd.numpy as npa
mp.verbosity(2)

import warnings
warnings.filterwarnings('ignore')
import io

from mpi4py import MPI
comm = MPI.COMM_WORLD

class custom_objective:
    def __init__(self,
                 lam_min, # in um
                 lam_max, # in um
                 design_dim, # in um
                 Nx,
                 Ny,
                 N_freq,
                 waveguide_width, # in um
                 mat_padding,
                 mat_waveguide,
                 bandpass_min,
                 bandpass_max,
                 ):
                 
        # Source
        self.lam_min = lam_min
        self.lam_max = lam_max
        
        # Geometries
        self.dy_design, self.dx_design = design_dim
        self.dx_pix = self.dx_design/Ny
        self.dy_pix = self.dy_design/Nx
        self.Nx = Ny
        self.Ny = Nx
        self.waveguide_width = waveguide_width
        
        # Set Bandpass Target Spectrum
        self.N_freq = N_freq
        self.wavelengths = np.linspace(self.lam_min, self.lam_max, self.N_freq)
        self.bandpass_target = np.zeros(self.N_freq)
        self.bandpass_target[(self.wavelengths >= bandpass_min)*(self.wavelengths <= bandpass_max)] = 1
        
        # Load Material Refractive Indices
        mat_type = np.array([mat_padding, mat_waveguide])
        raw_wavelength, mat_dict = rmd.load_all(self.wavelengths*1e3, 'n_k', mat_type)
        self.n_padding = np.mean(np.real(mat_dict[mat_padding]))
        self.n_waveguide = np.mean(np.real(mat_dict[mat_waveguide]))
        self.padding = mp.Medium(index=self.n_padding)
        self.waveguide = mp.Medium(index=self.n_waveguide)

        # Other Simulation Settings
        self.pml_size = 1.0
        
    def set_accuracy(self, upsampling_ratio):
        design_region_resolution = upsampling_ratio/self.dx_pix
        
        # Dimensions
        Sx = 2*self.pml_size + 2*self.lam_max + self.dx_design
        Sy = 2*self.pml_size + 1*self.lam_max + self.dy_design
        Sx = round(Sx*design_region_resolution)/design_region_resolution
        Sy = round(Sy*design_region_resolution)/design_region_resolution

        sim_size = mp.Vector3(Sx, Sy)
        #if comm.rank == 0:
            #print('\n' + str(design_region_resolution), flush=True)
    #        print((2*self.pml_size + 4*self.lam_max + self.dx_design), flush=True)
    #        print((2*self.pml_size + 2*self.lam_max + self.dy_design), flush=True)
    #        print(S_Nx, flush=True)
    #        print(S_Ny, flush=True)
            #print(Sx, flush=True)
            #print(Sy, flush=True)
            #print(Sx*design_region_resolution, flush=True)
            #print(Sy*design_region_resolution, flush=True)
        
        # PML
        pml_layers = [mp.PML(self.pml_size)]
        
        # Sources
        f_center = (1/self.lam_min + 1/self.lam_max)/2
        f_width = 2*np.max((np.abs(f_center - 1/self.lam_min),
                              np.abs(f_center - 1/self.lam_max)))
        source_center = [-Sx/2 + self.pml_size + 0.75*self.lam_max, 0, 0]
        source_size = mp.Vector3(0, self.dy_design, 0)
        kpoint = mp.Vector3(1, 0, 0)
        GaussianSrc = mp.GaussianSource(frequency=f_center, fwidth=f_width)
        sources = [mp.EigenModeSource(
            GaussianSrc,
            component=mp.Ez,
            eig_band=1,
            direction=mp.NO_DIRECTION,
            eig_kpoint=kpoint,
            size=source_size,
            center=source_center,
        )]
        
        # Symmetries
        symmetries = [mp.Mirror(direction=mp.Y)]
        
        # Design Region
        design_grid = mp.MaterialGrid(mp.Vector3(self.Nx, self.Ny), self.padding, self.waveguide)
        design_region = mpa.DesignRegion(
            design_grid,
            volume=mp.Volume(
                center=mp.Vector3(),
                size=mp.Vector3(self.dx_design, self.dy_design, 0),
            ),
        )
                                         
        # Geometries
        geometries = [
            mp.Block(
                center=mp.Vector3(x=-Sx/2 + self.pml_size/2 + 0.5*self.lam_max),
                material=self.waveguide,
                size=mp.Vector3(self.pml_size + self.lam_max, self.waveguide_width, 0)),
            mp.Block(
                center=mp.Vector3(x=Sx/2 - self.pml_size/2 - 0.5*self.lam_max),
                material=self.waveguide,
                size=mp.Vector3(self.pml_size + self.lam_max, self.waveguide_width, 0)),
            mp.Block(
                center=design_region.center,
                size=design_region.size,
                material=design_grid),
        ]
        
        # Create Simulation & Optimization Objects
        sim = mp.Simulation(
            cell_size=sim_size,
            boundary_layers=pml_layers,
            geometry=geometries,
            sources=sources,
            symmetries=symmetries,
            default_material=self.padding,
            resolution=design_region_resolution,
            force_all_components=True,
        )
        
        # Objective Functions
        mode_number = 1
        S0 = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(
                center=mp.Vector3(x=-Sx/2 + self.pml_size + 0.75*self.lam_max),
                size=mp.Vector3(0, self.dy_design, 0),
            ),
            mode_number,
        )
        
        S11 = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(
                center=mp.Vector3(x=-Sx/2 + self.pml_size + 0.5*self.lam_max),
                size=mp.Vector3(0, self.dy_design, 0),
            ),
            mode_number,
        )
        
        S21 = mpa.EigenmodeCoefficient(
            sim,
            mp.Volume(
                center=mp.Vector3(x=Sx/2 - self.pml_size - 0.5*self.lam_max),
                size=mp.Vector3(0, self.dy_design, 0),
            ),
            mode_number,
        )
        
        obj_list = [S0, S11, S21]
        
        self.opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=self.cost,
            objective_arguments=obj_list,
            design_regions=[design_region],
            frequencies=1/self.wavelengths,
            decay_by=1e-5)

        if comm.rank == 0:
            # Plot Simulation (debugging)
            self.opt.plot2D(True)
            plt.savefig(directory + '/simulation_geometry')
            plt.close()
    
    def cost(self, S0, S11, S21):
        T11 = npa.abs(S11/S0)**2
        T21 = npa.abs(S21/S0)**2
        cost = -(npa.mean(self.bandpass_target*T21) + npa.mean((1 - self.bandpass_target)*T11))/2
        #cost = -np.min((np.min(self.bandpass_target*T21), np.min((1 - self.bandpass_target)*T11)))
        
#        if comm.rank == 0:
#            print('T11: ' + str(T11), flush=True)
#            print('T21: ' + str(T21), flush=True)
        
        return cost
    
    def get_cost(self, x, get_grad=False):
        text_trap = io.StringIO()
        sys.stdout = text_trap
    
        x = x.reshape(self.Ny, self.Nx)
        x = np.rot90(x)
        x = x.reshape(-1)
        cost, jac = self.opt([x], need_gradient=get_grad)
        
        # Plot Simulation (debugging)
#        fig, ax = plt.subplots()
#        self.opt.plot2D(ax=ax, fields=mp.Ez, plot_eps_flag=True,
#                        field_parameters={"cmap": "RdBu", "alpha": 0.8})
#        plt.savefig(directory + '/simulated_fields')
#        plt.close()
        
        sys.stdout = sys.__stdout__
        
        if get_grad:
            jac = np.sum(jac, axis=1)
            
            return cost, jac
        else:
            return cost