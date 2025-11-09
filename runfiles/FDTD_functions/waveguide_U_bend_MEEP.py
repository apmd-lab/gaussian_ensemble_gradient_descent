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
mp.verbosity(0)

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
                 dimension=2,
                 ):
    
        self.dimension = dimension
    
        # Source
        self.lam_min = lam_min
        self.lam_max = lam_max
        
        # Geometries
        if self.dimension == 2:
            self.dy_design, self.dx_design = design_dim
        elif self.dimension == 3:
            self.dy_design, self.dx_design, self.dz_design = design_dim
        self.dx_pix = self.dx_design/Ny
        self.dy_pix = self.dy_design/Nx
        self.Nx = Ny
        self.Ny = Nx
        self.waveguide_width = waveguide_width
        
        # Set Wavelengths
        self.N_freq = N_freq
        self.wavelengths = np.linspace(self.lam_min, self.lam_max, self.N_freq)
        
        # Load Material Refractive Indices
        mat_type = np.array([mat_padding, mat_waveguide])
        raw_wavelength, mat_dict = rmd.load_all(self.wavelengths*1e3, 'n_k', mat_type)
        self.n_padding = np.mean(np.real(mat_dict[mat_padding]))
        self.n_waveguide = np.mean(np.real(mat_dict[mat_waveguide]))
        self.padding = mp.Medium(index=self.n_padding)
        self.waveguide = mp.Medium(index=self.n_waveguide)

        # Other Simulation Settings
        self.pml_size = self.lam_max
        
    def set_accuracy(self, upsampling_ratio):
        design_region_resolution = upsampling_ratio/self.dx_pix
        
        # Dimensions
        Sx = 2*self.pml_size + 2*self.lam_max + self.dx_design
        Sy = 2*self.pml_size + 1*self.lam_max + self.dy_design
        Sx = round(Sx*design_region_resolution)/design_region_resolution
        Sy = round(Sy*design_region_resolution)/design_region_resolution
        
        if self.dimension == 2:
            sim_size = mp.Vector3(Sx, Sy)
            
        elif self.dimension == 3:
            Sz = 2*self.pml_size + 1*self.lam_max + self.dz_design
    
            sim_size = mp.Vector3(Sx, Sy, Sz)
        
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
        f_width = 2*np.max((
            np.abs(f_center - 1/self.lam_min),
            np.abs(f_center - 1/self.lam_max),
        ))
        source_center = [-Sx/2 + self.pml_size + 0.5*self.lam_max, self.waveguide_width, 0]
        if self.dimension == 2:
            source_size = mp.Vector3(0, 1.5*self.waveguide_width, 0)
        elif self.dimension == 3:
            source_size = mp.Vector3(0, 1.5*self.waveguide_width, 1.5*self.dz_design)
        kpoint = mp.Vector3(1, 0, 0)
        GaussianSrc = mp.GaussianSource(frequency=f_center, fwidth=f_width)
        sources = [mp.EigenModeSource(
            GaussianSrc,
            component=mp.Ez,
            eig_band=1,
            direction=mp.NO_DIRECTION,
            eig_kpoint=kpoint,
            eig_parity=mp.EVEN_Y + mp.ODD_Z,
            size=source_size,
            center=source_center,
        )]
        
        # Symmetries
        if self.dimension == 2:
            symmetries = []
            
        if self.dimension == 3:
            symmetries = [
                mp.Mirror(direction=mp.Z, phase=-1),
            ]
        
        # Design Region
        if self.dimension == 2:
            design_grid = mp.MaterialGrid(mp.Vector3(self.Nx, self.Ny), self.padding, self.waveguide)
            design_region = mpa.DesignRegion(
                design_grid,
                volume=mp.Volume(
                    center=mp.Vector3(),
                    size=mp.Vector3(self.dx_design, self.dy_design),
                ),
            )
            
        elif self.dimension == 3:
            design_grid = mp.MaterialGrid(mp.Vector3(self.Nx, self.Ny, 0), self.padding, self.waveguide)
            design_region = mpa.DesignRegion(
                design_grid,
                volume=mp.Volume(
                    center=mp.Vector3(),
                    size=mp.Vector3(self.dx_design, self.dy_design, self.dz_design),
                ),
            )
                                         
        # Geometries (for reference simulation)
        if self.dimension == 2:
            geometries = [
                mp.Block(
                    center=mp.Vector3(
                        x=-Sx/2 + self.pml_size/2 + 0.5*self.lam_max,
                        y=self.waveguide_width),
                    material=self.waveguide,
                    size=mp.Vector3(self.pml_size + self.lam_max, self.waveguide_width, 0)),
                mp.Block(
                    center=mp.Vector3(
                        x=-Sx/2 + self.pml_size/2 + 0.5*self.lam_max,
                        y=-self.waveguide_width),
                    material=self.waveguide,
                    size=mp.Vector3(self.pml_size + self.lam_max, self.waveguide_width, 0)),
                mp.Block(
                    center=design_region.center,
                    size=design_region.size,
                    material=design_grid),
            ]
        
        elif self.dimension == 3:
            geometries = [
                mp.Block(
                    center=mp.Vector3(
                        x=-Sx/2 + self.pml_size/2 + 0.5*self.lam_max,
                        y=self.waveguide_width),
                    material=self.waveguide,
                    size=mp.Vector3(self.pml_size + self.lam_max, self.waveguide_width, self.dz_design)),
                mp.Block(
                    center=mp.Vector3(
                        x=-Sx/2 + self.pml_size/2 + 0.5*self.lam_max,
                        y=-self.waveguide_width),
                    material=self.waveguide,
                    size=mp.Vector3(self.pml_size + self.lam_max, self.waveguide_width, self.dz_design)),
                mp.Block(
                    center=design_region.center,
                    size=design_region.size,
                    material=design_grid),
            ]
        
        # Create Simulation & Optimization Objects
        self.sim = mp.Simulation(
            cell_size=sim_size,
            boundary_layers=pml_layers,
            geometry=geometries,
            sources=sources,
            symmetries=symmetries,
            default_material=self.padding,
            resolution=design_region_resolution,
            force_all_components=True,
        )
        
        # Check Mode Profile
#        sim.run(until=1)
#        
#        Ex = sim.get_array(center=source_center, size=source_size, component=mp.Ex)
#        Ey = sim.get_array(center=source_center, size=source_size, component=mp.Ey)
#        Ez = sim.get_array(center=source_center, size=source_size, component=mp.Ez)
#        
#        if comm.rank == 0:
#            np.savez(directory + '/mode_profile', Ex=Ex, Ey=Ey, Ez=Ez)
#        assert False
                
        # Objective Functions
        mode_number = 1
        if self.dimension == 2:
            S0 = mpa.EigenmodeCoefficient(
                self.sim,
                mp.Volume(
                    center=mp.Vector3(
                        x=-Sx/2 + self.pml_size + 0.6*self.lam_max,
                        y=self.waveguide_width),
                    size=mp.Vector3(0, 1.5*self.waveguide_width, 0),
                ),
                mode_number,
                forward=True,
            )
            
            S21 = mpa.EigenmodeCoefficient(
                self.sim,
                mp.Volume(
                    center=mp.Vector3(
                        x=-Sx/2 + self.pml_size + 0.5*self.lam_max,
                        y=-self.waveguide_width),
                    size=mp.Vector3(0, 1.5*self.waveguide_width, 0),
                ),
                mode_number,
                forward=False,
            )
        
        elif self.dimension == 3:
            S0 = mpa.EigenmodeCoefficient(
                self.sim,
                mp.Volume(
                    center=mp.Vector3(
                        x=-Sx/2 + self.pml_size + 0.5*self.lam_max,
                        y=self.waveguide_width),
                    size=mp.Vector3(0, 1.5*self.waveguide_width, 1.5*self.dz_design),
                ),
                mode_number,
                forward=True,
            )
            
            S21 = mpa.EigenmodeCoefficient(
                self.sim,
                mp.Volume(
                    center=mp.Vector3(
                        x=-Sx/2 + self.pml_size + 0.5*self.lam_max,
                        y=-self.waveguide_width),
                    size=mp.Vector3(0, 1.5*self.waveguide_width, 1.5*self.dz_design),
                ),
                mode_number,
                forward=False,
            )
        
        obj_list = [S0, S21]
        
        self.opt = mpa.OptimizationProblem(
            simulation=self.sim,
            objective_functions=self.cost,
            objective_arguments=obj_list,
            design_regions=[design_region],
            frequencies=1/self.wavelengths,
            maximum_run_time=1000)

#        if comm.rank == 0:
#            # Plot Simulation (debugging)
#            self.opt.plot2D(True, output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(Sx, Sy, 0)))
#            plt.savefig(directory + '/simulation_geometry')
#            plt.close()
#        assert False
    
    def cost(self, S0, S21):
        T21 = npa.abs(S21/S0)**2
        cost = -npa.mean(T21)
        
        #cost = -npa.mean(npa.abs(S21)**2)
        
        return cost
    
    def get_cost(self, x, get_grad=False):
        text_trap_out = io.StringIO()
        text_trap_err = io.StringIO()
        sys.stdout = text_trap_out
        sys.stderr = text_trap_err
    
        x_temp = x.copy()
    
        x = x.reshape(self.Ny, self.Nx)
        x = np.rot90(x, k=-1)
        x = x.reshape(-1)
        
        cost, jac = self.opt([x], need_gradient=get_grad)
        
        if get_grad:
            jac = np.sum(jac, axis=1)
            jac = jac.reshape(self.Nx, self.Ny)
            jac = np.rot90(jac, k=1)
            jac = jac.reshape(-1)
        
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
#        eps = self.sim.get_array(center=mp.Vector3(), size=mp.Vector3(self.dx_design, self.dy_design), component=mp.Dielectric)
#        np.savez(directory + '/debug_eps', x=x.reshape(self.Nx, self.Ny), eps=eps, n_padding=self.n_padding, n_waveguide=self.n_waveguide)
#        assert False
        
#        if get_grad:
#            np.savez(directory + '/debug_jac', x=x_temp, jac=jac)
#            assert False
        
        # Plot Simulation (debugging)
#        fig, ax = plt.subplots()
#        self.opt.plot2D(ax=ax, fields=mp.Ez, plot_eps_flag=True,
#                        field_parameters={"cmap": "RdBu", "alpha": 0.8})
#        plt.savefig(directory + '/simulated_fields')
#        plt.close()
        #assert False
        
        if get_grad:
            return cost, jac
        else:
            return cost