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
                 lam_tgt, # in um
                 design_dim, # in um
                 Nx,
                 Ny,
                 waveguide_width, # in um
                 mat_padding,
                 mat_waveguide,
                 dimension=2,
                 ):
    
        self.dimension = dimension
    
        # Source
        self.lam_tgt = lam_tgt
        
        # Geometries
        if self.dimension == 2:
            self.dx_design, self.dy_design = design_dim
        elif self.dimension == 3:
            self.dx_design, self.dy_design, self.dz_design = design_dim
        self.dx_pix = self.dx_design/Nx
        self.dy_pix = self.dy_design/Ny
        self.Nx = Nx
        self.Ny = Ny
        self.waveguide_width = waveguide_width
        
        # Load Material Refractive Indices
        mat_type = np.array([mat_padding, mat_waveguide])
        raw_wavelength, mat_dict = rmd.load_all(np.array([self.lam_tgt*1e3]), 'n_k', mat_type)
        self.n_padding = np.mean(np.real(mat_dict[mat_padding][0]))
        self.n_waveguide = np.mean(np.real(mat_dict[mat_waveguide][0]))
        self.padding = mp.Medium(index=self.n_padding)
        self.waveguide = mp.Medium(index=self.n_waveguide)

        # Other Simulation Settings
        self.pml_size = self.lam_tgt
        self.P0_dict = dict()
        
    def set_accuracy(self, upsampling_ratio):
        self.design_region_resolution = upsampling_ratio/self.dx_pix
        
        # Dimensions
        self.Sx = 2*self.pml_size + 2*self.lam_tgt + self.dx_design
        self.Sy = 2*self.pml_size + 1*self.lam_tgt + self.dy_design
        self.Sx = round(self.Sx*self.design_region_resolution)/self.design_region_resolution
        self.Sy = round(self.Sy*self.design_region_resolution)/self.design_region_resolution
        
        if self.dimension == 2:
            self.sim_size = mp.Vector3(self.Sx, self.Sy)
            
        elif self.dimension == 3:
            Sz = 2*self.pml_size + 1*self.lam_tgt + self.dz_design
    
            self.sim_size = mp.Vector3(self.Sx, self.Sy, Sz)
        
        # PML
        self.pml_layers = [mp.PML(self.pml_size)]
        
        # Sources
        f_center = 1/self.lam_tgt
        f_width = f_center/2
        source_center = [-self.Sx/2 + self.pml_size + 0.5*self.lam_tgt, 0, 0]
        if self.dimension == 2:
            source_size = mp.Vector3(0, 1.5*self.waveguide_width, 0)
        elif self.dimension == 3:
            source_size = mp.Vector3(0, 1.5*self.waveguide_width, 1.5*self.dz_design)
        kpoint = mp.Vector3(1, 0, 0)
        GaussianSrc = mp.GaussianSource(frequency=f_center, fwidth=f_width)
        ContinuousSrc = mp.ContinuousSource(frequency=f_center, width=20)
        self.sources = [mp.EigenModeSource(
            GaussianSrc,
            component=mp.Ez,
            eig_band=1,
            direction=mp.NO_DIRECTION,
            eig_kpoint=kpoint,
            eig_parity=mp.EVEN_Y, # + mp.ODD_Z,
            size=source_size,
            center=source_center,
        )]
        
        # Symmetries
        if self.dimension == 2:
            self.symmetries = []
            
        if self.dimension == 3:
            self.symmetries = [
                mp.Mirror(direction=mp.Z, phase=-1),
            ]
        
        # Design Region
        if self.dimension == 2:
            design_grid = mp.MaterialGrid(mp.Vector3(self.Nx, self.Ny), self.padding, self.waveguide)
            self.design_region = mpa.DesignRegion(
                design_grid,
                volume=mp.Volume(
                    center=mp.Vector3(),
                    size=mp.Vector3(self.dx_design, self.dy_design),
                ),
            )
            
        elif self.dimension == 3:
            design_grid = mp.MaterialGrid(mp.Vector3(self.Nx, self.Ny, 0), self.padding, self.waveguide)
            self.design_region = mpa.DesignRegion(
                design_grid,
                volume=mp.Volume(
                    center=mp.Vector3(),
                    size=mp.Vector3(self.dx_design, self.dy_design, self.dz_design),
                ),
            )
        
        # Geometries
        if self.dimension == 2:
            geometries_norm = [
                mp.Block(
                    center=mp.Vector3(),
                    material=self.waveguide,
                    size=mp.Vector3(self.Sx, self.waveguide_width, 0),
                ),
            ]
            
            self.geometries = [
                mp.Block(
                    center=mp.Vector3(
                        x=-self.Sx/2 + self.pml_size/2 + 0.5*self.lam_tgt,
                        y=0.0),
                    material=self.waveguide,
                    size=mp.Vector3(self.pml_size + self.lam_tgt, self.waveguide_width, 0)),
                mp.Block(
                    center=mp.Vector3(
                        x=self.Sx/2 - self.pml_size/2 - 0.5*self.lam_tgt,
                        y=0.0),
                    material=self.waveguide,
                    size=mp.Vector3(self.pml_size + self.lam_tgt, self.waveguide_width, 0)),
                mp.Block(
                    center=self.design_region.center,
                    size=self.design_region.size,
                    material=design_grid),
            ]
        
        elif self.dimension == 3:
            geometries_norm = [
                mp.Block(
                    center=mp.Vector3(),
                    material=self.waveguide,
                    size=mp.Vector3(self.Sx, self.waveguide_width, self.dz_design),
                ),
            ]
        
            self.geometries = [
                mp.Block(
                    center=mp.Vector3(
                        x=-self.Sx/2 + self.pml_size/2 + 0.5*self.lam_tgt,
                        y=0.0),
                    material=self.waveguide,
                    size=mp.Vector3(self.pml_size + self.lam_tgt, self.waveguide_width, self.dz_design)),
                mp.Block(
                    center=mp.Vector3(
                        x=self.Sx/2 - self.pml_size/2 - 0.5*self.lam_tgt,
                        y=0.0),
                    material=self.waveguide,
                    size=mp.Vector3(self.pml_size + self.lam_tgt, self.waveguide_width, self.dz_design)),
                mp.Block(
                    center=self.design_region.center,
                    size=self.design_region.size,
                    material=design_grid),
            ]
        
        if not upsampling_ratio in self.P0_dict.keys():
            # Normalization Run
            sim = mp.Simulation(
                cell_size=self.sim_size,
                boundary_layers=self.pml_layers,
                geometry=geometries_norm,
                sources=self.sources,
                symmetries=self.symmetries,
                default_material=self.padding,
                resolution=self.design_region_resolution,
                force_all_components=True,
            )
            
            source_monitor = sim.add_mode_monitor(
                f_center,
                f_width,
                1,
                mp.ModeRegion(
                    center=mp.Vector3(
                        x=-self.Sx/2 + self.pml_size + 0.6*self.lam_tgt,
                        y=0.0,
                    ),
                    size=mp.Vector3(0, 1.5*self.waveguide_width, 0),
                ),
                yee_grid=True,
            )
            
            sim.run(
                until_after_sources=mp.stop_when_fields_decayed(
                    50,
                    mp.Ez,
                    mp.Vector3(
                        x=-self.Sx/2 + self.pml_size + 0.6*self.lam_tgt,
                        y=0.0,
                    ),
                    1e-11,
                ),
            )
            
            result_norm = sim.get_eigenmode_coefficients(
                source_monitor,
                [1],
                eig_parity=mp.EVEN_Y, # + mp.ODD_Z,
            )
            
            self.P0_dict[upsampling_ratio] = np.abs(result_norm.alpha[0,:,0])**2
        
        self.P0 = self.P0_dict[upsampling_ratio]
        
#        fig, ax = plt.subplots()
#        self.sim.plot2D(ax=ax, fields=mp.Ez, plot_eps_flag=True,
#                        field_parameters={"cmap": "RdBu", "alpha": 0.8})
#        plt.savefig(directory + '/simulated_fields')
#        plt.close()
#        assert False
        
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

#        if comm.rank == 0:
#            # Plot Simulation (debugging)
#            self.opt.plot2D(True, output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(Sx, Sy, 0)))
#            plt.savefig(directory + '/simulation_geometry')
#            plt.close()
#        assert False
    
    def cost(self, S21):
        cost = npa.log(npa.abs(S21)**2/self.P0)/npa.log(1e-27) - 1
#        if comm.rank == 0:
#            print(self.P0, flush=True)
#            print(cost, flush=True)
        
        #cost = -1e27*npa.abs(S21)**2
        
        return cost
    
    def get_cost(self, x, get_grad=False, print_messages=False, print_errors=False):
        if not print_messages:
            text_trap_out = io.StringIO()
            sys.stdout = text_trap_out
        if not print_errors:
            text_trap_err = io.StringIO()
            sys.stderr = text_trap_err
    
        x = x.reshape(-1)
        
        # Create Simulation & Optimization Objects
        sim = mp.Simulation(
            cell_size=self.sim_size,
            boundary_layers=self.pml_layers,
            geometry=self.geometries,
            sources=self.sources,
            symmetries=self.symmetries,
            default_material=self.padding,
            resolution=self.design_region_resolution,
            force_all_components=True,
        )
        
        # Objective Functions
        if self.dimension == 2:
            S21 = mpa.EigenmodeCoefficient(
                sim,
                mp.Volume(
                    center=mp.Vector3(
                        x=self.Sx/2 - self.pml_size - 0.5*self.lam_tgt,
                        y=0.0),
                    size=mp.Vector3(0, 1.5*self.waveguide_width, 0),
                ),
                2,
                forward=True,
            )
        
        elif self.dimension == 3:
            S21 = mpa.EigenmodeCoefficient(
                sim,
                mp.Volume(
                    center=mp.Vector3(
                        x=self.Sx/2 - self.pml_size - 0.5*self.lam_tgt,
                        y=0.0),
                    size=mp.Vector3(0, 1.5*self.waveguide_width, 1.5*self.dz_design),
                ),
                2,
                forward=False,
            )
        
        obj_list = [S21]
        
        opt = mpa.OptimizationProblem(
            simulation=sim,
            objective_functions=self.cost,
            objective_arguments=obj_list,
            design_regions=[self.design_region],
            frequencies=[1/self.lam_tgt],
            decay_by=1e-12,
            #maximum_run_time=1000,
        )
        
        cost, jac = opt([x], need_gradient=get_grad)
        
        jac_FD, idx = opt.calculate_fd_gradient(num_gradients=10, db=1e-6)
        
        if get_grad:
            jac = jac.reshape(self.Nx, self.Ny)
        
        if not print_messages:
            sys.stdout = sys.__stdout__
        if not print_errors:
            sys.stderr = sys.__stderr__
        
#        eps = sim.get_array(center=mp.Vector3(), size=mp.Vector3(self.dx_design, self.dy_design), component=mp.Dielectric)
#        np.savez(directory + '/debug_eps', x=x.reshape(self.Nx, self.Ny), eps=eps, n_padding=self.n_padding, n_waveguide=self.n_waveguide)
#        assert False
        
        if get_grad:
            if comm.rank == 0:
                jac=jac.reshape(-1)
                np.savez(directory + '/debug_jac', x=x, jac=jac, jac_FD=jac_FD, idx=idx)
                
                fig = plt.figure(figsize = (8,5))
                ax  = fig.add_subplot(1,1,1)
                ax.plot(jac_FD,jac[idx],"ro")
                jac_min = np.min((np.min(jac_FD), np.min(jac[idx])))
                jac_max = np.max((np.max(jac_FD), np.max(jac[idx])))
                ax.plot(np.linspace(jac_min, jac_max, 100), np.linspace(jac_min, jac_max, 100), '--', label='y=x')
                ax.set_xlabel("Finite-difference gradient")
                ax.set_ylabel("Adjoint gradient")
                fig.savefig(f'Finite-difference-adjoint-comparison.png',  dpi=fig.dpi, transparent = False, facecolor='white')
            assert False
        
        # Plot Simulation (debugging)
#        fig, ax = plt.subplots()
#        opt.plot2D(ax=ax, fields=mp.Ez, plot_eps_flag=True,
#                        field_parameters={"cmap": "RdBu", "alpha": 0.8})
#        plt.savefig(directory + '/simulated_fields')
#        plt.close()
        #assert False
        
        if get_grad:
            return cost, jac
        else:
            return cost