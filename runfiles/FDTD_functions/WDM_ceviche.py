import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles')

import numpy as np
from scipy.ndimage import zoom, gaussian_filter
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

c = 299792458 # 1.0 #299792458
um = 1e-6 # 1.0 #1e-6
nm = 1e3

class custom_objective:
    def __init__(
        self,
        lam_tgt, # in um
        design_dim, # in um
        Nx,
        Ny,
        waveguide_width, # in um
        mat_padding,
        mat_waveguide,
        padding,
        min_feature_size,
        IPR_exponent,
    ):
    
        # Source
        self.lam_tgt = lam_tgt*um
        
        # Geometries
        self.dx_design, self.dy_design = design_dim*um
        self.dx_pix = self.dx_design/Nx
        self.Nx = Nx
        self.Ny = Ny
        self.waveguide_width = waveguide_width*um
        self.padding = padding
        self.min_feature_size = min_feature_size
        
        # Load Material Refractive Indices
        mat_type = np.array([mat_padding, mat_waveguide])
        raw_wavelength, mat_dict = rmd.load_all(lam_tgt*nm, 'n_k', mat_type)
        self.n_padding = np.mean(np.real(mat_dict[mat_padding]))
        self.n_waveguide = np.mean(np.real(mat_dict[mat_waveguide]))

        # Other Simulation Settings
        self.pml_size = np.max(self.lam_tgt)

        # Cost Function Settings
        self.IPR_exponent = IPR_exponent

    def set_accuracy(self, upsampling_ratio):
        self.upsampling_ratio = upsampling_ratio
        self.resolution = self.upsampling_ratio/self.dx_pix
        
        ### Dimensions
        self.Npml = int(self.pml_size*self.resolution)
        self.Nlam_half = int(np.max(self.lam_tgt) / 2 *self.resolution)
        self.Nx_up = int(self.Nx*self.upsampling_ratio)
        self.Ny_up = int(self.Ny*self.upsampling_ratio)
        
        Sx = 2*self.Npml + 4*self.Nlam_half + self.Nx_up
        Sy = 2*self.Npml + 2*self.Nlam_half + self.Ny_up
        
        center_y = int(np.floor(Sy/2 + 0.5))
        
        waveguide_width_npix = np.round(self.waveguide_width*self.resolution)
        if self.Nx % 2 == 0:
            waveguide_width_npix += (waveguide_width_npix % 2)*np.sign(self.waveguide_width*self.resolution - waveguide_width_npix)
        else:
            waveguide_width_npix += (waveguide_width_npix % 2 == 0)*np.sign(self.waveguide_width*self.resolution - waveguide_width_npix)
        waveguide_halfwidth_npix = int(np.floor(waveguide_width_npix/2 + 0.5))
        
        # Input Waveguide (Top)
        waveguide_bottom_input = int(center_y - waveguide_halfwidth_npix)
        waveguide_top_input = int(waveguide_bottom_input + waveguide_width_npix)
        
        # Output Waveguides (Offset)
        waveguide_bottom_output1 = int(center_y + 2 * waveguide_width_npix - waveguide_halfwidth_npix)
        waveguide_top_output1 = int(waveguide_bottom_output1 + waveguide_width_npix)

        waveguide_bottom_output2 = int(center_y - waveguide_halfwidth_npix)
        waveguide_top_output2 = int(waveguide_bottom_output2 + waveguide_width_npix)
        
        waveguide_bottom_output3 = int(center_y - 2 * waveguide_width_npix - waveguide_halfwidth_npix)
        waveguide_top_output3 = int(waveguide_bottom_output3 + waveguide_width_npix)

        #---------------------------------------------------------------------------------------------------
        # Straight Waveguide Simulation for Normalization
        #---------------------------------------------------------------------------------------------------
        density_normalization = np.zeros((Sx, Sy))
        density_normalization[:,waveguide_bottom_input:waveguide_top_input] = 1

        # Source (Input)
        self.omega = 2*np.pi*c/self.lam_tgt
        
        self.source_plane = Slice(
            x = np.array([int(self.Npml + self.Nlam_half)]),
            y = np.arange(waveguide_bottom_input - waveguide_halfwidth_npix, waveguide_top_input + waveguide_halfwidth_npix).astype(np.int32),
        )
        
        # Use background eps (waveguides only) for defining modes
        eps_normalization = ((self.n_waveguide - self.n_padding)*density_normalization + self.n_padding)**2
        
        self.P_norm = np.zeros(self.lam_tgt.size)
        for i in range(self.lam_tgt.size):
            source = insert_mode(
                self.omega[i], 1/self.resolution, self.source_plane.x, self.source_plane.y, eps_normalization, m=1,
            )
        
            # Monitors (Output) -- CENTERED for Normalization
            self.monitor_plane = Slice(
                x = np.array([int(Sx - self.Npml - self.Nlam_half)]),
                y = np.arange(waveguide_bottom_input - waveguide_halfwidth_npix, waveguide_top_input + waveguide_halfwidth_npix).astype(np.int32),
            )
            # Use same mode profile as source (Fundamental TM0) at output
            monitor_norm = insert_mode(
                self.omega[i], 1/self.resolution, self.monitor_plane.x, self.monitor_plane.y, eps_normalization, m=1,
            )

            simulation = fdfd_ez(self.omega[i], 1/self.resolution, eps_normalization, [self.Npml, self.Npml])
            _, _, Ez_norm = simulation.solve(source)
            
            # Calculate Overlap Normalization (Power coupling into fundamental mode of straight waveguide)
            self.P_norm[i] = npa.abs(npa.sum(npa.conj(Ez_norm[self.monitor_plane.x, self.monitor_plane.y])\
                                            * monitor_norm[self.monitor_plane.x, self.monitor_plane.y]))
        
        #self.visualize(Ez_TM0_norm, eps_normalization, 'TM0_normalization')

        #---------------------------------------------------------------------------------------------------
        # Simulation with Freeform Design Region
        #---------------------------------------------------------------------------------------------------

        ### Geometries
        self.density_background = np.zeros((Sx, Sy))
        self.mask_design = np.zeros((Sx, Sy))
        density_design = np.zeros((Sx, Sy))
        
        # Input Waveguide (Top)
        self.density_background[
            :self.Npml+2*self.Nlam_half,
            waveguide_bottom_input:waveguide_top_input,
        ] = 1
        
        # Output Waveguide Top
        self.density_background[
            -self.Npml-2*self.Nlam_half:,
            waveguide_bottom_output1:waveguide_top_output1,
        ] = 1
        
        # Output Waveguide Center
        self.density_background[
            -self.Npml-2*self.Nlam_half:,
            waveguide_bottom_output2:waveguide_top_output2,
        ] = 1
        
        # Output Waveguide Bottom
        self.density_background[
            -self.Npml-2*self.Nlam_half:,
            waveguide_bottom_output3:waveguide_top_output3,
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
        
        # Source (Input)
        self.omega = 2*np.pi*c/self.lam_tgt
        
        self.source_plane = Slice(
            x = np.array([int(self.Npml + self.Nlam_half)]),
            y = np.arange(waveguide_bottom_input - waveguide_halfwidth_npix, waveguide_top_input + waveguide_halfwidth_npix).astype(np.int32),
        )
        
        # Use background eps (waveguides only) for defining modes
        eps_bg = ((self.n_waveguide - self.n_padding)*self.density_background + self.n_padding)**2
        
        self.source1 = insert_mode(
            self.omega[0], 1/self.resolution, self.source_plane.x, self.source_plane.y, eps_bg, m=1,
        )
        self.source2 = insert_mode(
            self.omega[1], 1/self.resolution, self.source_plane.x, self.source_plane.y, eps_bg, m=1,
        )
        self.source3 = insert_mode(
            self.omega[2], 1/self.resolution, self.source_plane.x, self.source_plane.y, eps_bg, m=1,
        )
        
        # Monitors (Output)
        self.monitor_plane1 = Slice(
            x = np.array([int(Sx - self.Npml - self.Nlam_half)]),
            y = np.arange(waveguide_bottom_output1 - waveguide_halfwidth_npix, waveguide_top_output1 + waveguide_halfwidth_npix).astype(np.int32),
        )
        self.monitor_plane2 = Slice(
            x = np.array([int(Sx - self.Npml - self.Nlam_half)]),
            y = np.arange(waveguide_bottom_output2 - waveguide_halfwidth_npix, waveguide_top_output2 + waveguide_halfwidth_npix).astype(np.int32),
        )
        self.monitor_plane3 = Slice(
            x = np.array([int(Sx - self.Npml - self.Nlam_half)]),
            y = np.arange(waveguide_bottom_output3 - waveguide_halfwidth_npix, waveguide_top_output3 + waveguide_halfwidth_npix).astype(np.int32),
        )
        
        # Monitors use Fundamental Mode (m=1) at output waveguides
        self.monitor1 = insert_mode(
            self.omega[0], 1/self.resolution, self.monitor_plane1.x, self.monitor_plane1.y, eps_bg, m=1,
        )
        self.monitor2 = insert_mode(
            self.omega[1], 1/self.resolution, self.monitor_plane2.x, self.monitor_plane2.y, eps_bg, m=1,
        )
        self.monitor3 = insert_mode(
            self.omega[2], 1/self.resolution, self.monitor_plane3.x, self.monitor_plane3.y, eps_bg, m=1,
        )

    def upsample_design(self, x):
        x = x.reshape(self.Nx, self.Ny).astype(np.float64)
        x_padded = self.padding.copy()
        x_padded[self.min_feature_size:-self.min_feature_size, self.min_feature_size:-self.min_feature_size] = x

        if self.upsampling_ratio > 1:
            x_upsampled = np.where(gaussian_filter(zoom(x_padded, self.upsampling_ratio, order=1), self.upsampling_ratio, mode='constant') > 0.5, 1, 0)
        elif self.upsampling_ratio < 1:
            x_upsampled = zoom(x_padded, self.upsampling_ratio, order=0)
        else:
            x_upsampled = x_padded.copy()
        
        x_cropped = x_upsampled[
            int(self.min_feature_size*self.upsampling_ratio):-int(self.min_feature_size*self.upsampling_ratio),
            int(self.min_feature_size*self.upsampling_ratio):-int(self.min_feature_size*self.upsampling_ratio)
        ]

        return x_cropped

    def visualize(self, Ez, eps, suffix):
        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6,3))
        ceviche.viz.real(Ez, outline=eps, ax=ax[0], cbar=False)
        ax[0].plot(self.source_plane.x*np.ones(len(self.source_plane.y)), self.source_plane.y, 'r-')
        ax[0].plot(self.monitor_plane1.x*np.ones(len(self.monitor_plane1.y)), self.monitor_plane1.y, 'b-')
        try:
            ax[0].plot(self.monitor_plane2.x*np.ones(len(self.monitor_plane2.y)), self.monitor_plane2.y, 'b-')
            ax[0].plot(self.monitor_plane3.x*np.ones(len(self.monitor_plane3.y)), self.monitor_plane3.y, 'b-')
        except:
            pass
        ceviche.viz.abs(eps, ax=ax[1], cmap='Greys')
        plt.savefig('simulation_visualization' + suffix)
        plt.close()

    def cost(self, density_design):
        density_combined = density_design*self.mask_design + self.density_background*(self.mask_design==0).astype(npa.float64)
        eps = ((self.n_waveguide - self.n_padding)*density_combined + self.n_padding)**2
        
        # Simulation 1: Input Wvl 1 -> Should go to Top Output
        simulation1 = fdfd_ez(self.omega[0], 1/self.resolution, eps, [self.Npml, self.Npml])
        _, _, Ez1 = simulation1.solve(self.source1)
        
        # Overlap with Top Monitor (Maximize)
        # Using simple overlap integral squared as proxy for power transmission
        trans1 = npa.abs(npa.sum(npa.conj(Ez1[self.monitor_plane1.x, self.monitor_plane1.y]) \
                                        * self.monitor1[self.monitor_plane1.x, self.monitor_plane1.y])) \
                        / self.P_norm[0]

        #self.visualize(Ez1, eps, 'lam1_' + str(int(10*self.upsampling_ratio)))
        #print('trans_TM0_top: ', trans_TM0_top, flush=True)
        #i = 0
        #while os.path.exists('simulation_visualizationTM0_design' + str(i) + '.png'):
        #    i += 1
        #self.visualize(Ez_TM0, eps, 'TM0_design' + str(i))
        #np.savez('source_TM0_' + str(i) + '.npz', self.source_TM0, Ez_TM0)

        # Simulation 2: Input Wvl 2 -> Should go to Center Output
        simulation2 = fdfd_ez(self.omega[1], 1/self.resolution, eps, [self.Npml, self.Npml])
        _, _, Ez2 = simulation2.solve(self.source2)
        
        # Overlap with Center Monitor (Maximize)
        trans2 = npa.abs(npa.sum(npa.conj(Ez2[self.monitor_plane2.x, self.monitor_plane2.y]) \
                                        * self.monitor2[self.monitor_plane2.x, self.monitor_plane2.y])) \
                        / self.P_norm[1]
        
        #self.visualize(Ez2, eps, 'lam2_' + str(int(10*self.upsampling_ratio)))
        #print('trans_TM1_bot: ', trans_TM1_bot, flush=True)
        #i = 0
        #while os.path.exists('simulation_visualizationTM1_design' + str(i) + '.png'):
        #    i += 1
        #self.visualize(Ez_TM1, eps, 'TM1_design' + str(i))
        #np.savez('source_TM1_' + str(i) + '.npz', self.source_TM1, Ez_TM1)
        #if i > 40:
        #    assert False

        # Simulation 3: Input Wvl 3 -> Should go to Bot Output
        simulation3 = fdfd_ez(self.omega[2], 1/self.resolution, eps, [self.Npml, self.Npml])
        _, _, Ez3 = simulation3.solve(self.source3)
        
        # Overlap with Bot Monitor (Maximize)
        trans3 = npa.abs(npa.sum(npa.conj(Ez3[self.monitor_plane3.x, self.monitor_plane3.y]) \
                                        * self.monitor3[self.monitor_plane3.x, self.monitor_plane3.y])) \
                        / self.P_norm[2]
        
        #self.visualize(Ez3, eps, 'lam3_' + str(int(10*self.upsampling_ratio)))
        #print('trans_TM1_bot: ', trans_TM1_bot, flush=True)
        #i = 0
        #while os.path.exists('simulation_visualizationTM1_design' + str(i) + '.png'):
        #    i += 1
        #self.visualize(Ez_TM1, eps, 'TM1_design' + str(i))
        #np.savez('source_TM1_' + str(i) + '.npz', self.source_TM1, Ez_TM1)
        #if i > 40:
        #    assert False
        #assert False

        # Total Cost
        cost = -(trans1**self.IPR_exponent + trans2**self.IPR_exponent + trans3**self.IPR_exponent) / 3 + 0.3
        
        return cost
    
    def get_cost(self, x, get_grad=False):
        x_cropped = self.upsample_design(x)
        
        density_design = npa.zeros_like(self.density_background).astype(np.float64)
        x_length, y_length = density_design[
            self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
            self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
        ].shape
        if x_cropped.shape[0] == x_length and x_cropped.shape[1] == y_length:
            density_design[
                self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
                self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
            ] = x_cropped
        elif np.abs(x_cropped.shape[0] - x_length) == 1 and np.abs(x_cropped.shape[1] - y_length) == 1:
            density_design[
                self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
                self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
            ] = x_cropped[:-1,:-1]
        elif np.abs(x_cropped.shape[0] - x_length) == 2 and np.abs(x_cropped.shape[1] - y_length) == 2:
            density_design[
                self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
                self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
            ] = x_cropped[1:-1,1:-1]
        else:
            raise ValueError('x_cropped shape ' + str(x_cropped.shape) + ' is not compatible with density_design shape ' + str(density_design.shape))
        
        cost = self.cost(density_design)
        
        if get_grad:
            jac = jacobian(self.cost, mode='reverse')(density_design)
            jac = jac[0,:].reshape(density_design.shape)
            jac = jac[
                self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
                self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
            ]
            return cost, jac
        else:
            return cost
    
    def get_transmission_and_fields(self, x):
        x_cropped = self.upsample_design(x)
        
        density_design = npa.zeros_like(self.density_background).astype(np.float64)
        x_length, y_length = density_design[
            self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
            self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
        ].shape
        if x_cropped.shape[0] == x_length and x_cropped.shape[1] == y_length:
            density_design[
                self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
                self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
            ] = x_cropped
        elif np.abs(x_cropped.shape[0] - x_length) == 1 and np.abs(x_cropped.shape[1] - y_length) == 1:
            density_design[
                self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
                self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
            ] = x_cropped[:-1,:-1]
        elif np.abs(x_cropped.shape[0] - x_length) == 2 and np.abs(x_cropped.shape[1] - y_length) == 2:
            density_design[
                self.Npml+2*self.Nlam_half:-self.Npml-2*self.Nlam_half,
                self.Npml+self.Nlam_half:-self.Npml-self.Nlam_half,
            ] = x_cropped[1:-1,1:-1]
        else:
            raise ValueError('x_cropped shape ' + str(x_cropped.shape) + ' is not compatible with density_design shape ' + str(density_design.shape))
        
        eps = ((self.n_waveguide - self.n_padding)*density_design + self.n_padding)**2
        
        # Simulation 1: Input Wvl 1 -> Should go to Top Output
        Ez1 = self.run_simulation(eps, self.source1)
        
        # Overlap with Top Monitor (Maximize)
        # Using simple overlap integral squared as proxy for power transmission
        trans1 = npa.abs(npa.sum(npa.conj(Ez1[self.monitor_plane1.x, self.monitor_plane1.y]) \
                                        * self.monitor1[self.monitor_plane1.x, self.monitor_plane1.y])) \
                        / self.P_norm[0]
        
        # Simulation 2: Input Wvl 2 -> Should go to Center Output
        Ez2 = self.run_simulation(eps, self.source2)
        
        # Overlap with Center Monitor (Maximize)
        trans2 = npa.abs(npa.sum(npa.conj(Ez2[self.monitor_plane2.x, self.monitor_plane2.y]) \
                                        * self.monitor2[self.monitor_plane2.x, self.monitor_plane2.y])) \
                        / self.P_norm[1]
        
        # Simulation 3: Input Wvl 3 -> Should go to Bot Output
        Ez3 = self.run_simulation(eps, self.source3)
        
        # Overlap with Bot Monitor (Maximize)
        trans3 = npa.abs(npa.sum(npa.conj(Ez3[self.monitor_plane3.x, self.monitor_plane3.y]) \
                                        * self.monitor3[self.monitor_plane3.x, self.monitor_plane3.y])) \
                        / self.P_norm[2]
        
        return trans1, trans2, trans3, Ez1, Ez2, Ez3