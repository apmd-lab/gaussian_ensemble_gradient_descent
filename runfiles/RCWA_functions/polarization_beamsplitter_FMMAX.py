import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent/runfiles')

from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil
import time
import numpy as np
import jax
import jax.numpy as jnp
from fmmax import basis, fields, fmm, scattering
import util.read_mat_data as rmd

class custom_objective:
    def __init__(
        self,
        Nx,
        Ny,
        period,
        thickness_pattern,
        lam,
        in_plane_wavevector,
        mat_background,
        mat_pattern,
        diff_order,
        truncation=basis.Truncation.CIRCULAR,
        formulation=fmm.Formulation.JONES_DIRECT, #FFT, JONES_DIRECT, JONES, NORMAL, POL
        IPR_exponent=1/5,
    ):

        # Set Material Properties
        self.lam = jnp.asarray(lam)
        self.freq = jnp.asarray(1/lam)
        
        mat_type = list(set(np.hstack((mat_pattern, mat_background))))
        raw_wavelength, mat_dict = rmd.load_all(1e3*lam, 'n_k', mat_type)

        self.eps_incident_medium = jnp.asarray(mat_dict[mat_background[0]]**2)[:,jnp.newaxis,jnp.newaxis] # freq, x, y
        self.eps_substrate = jnp.asarray(mat_dict[mat_background[1]]**2)[:,jnp.newaxis,jnp.newaxis]
        self.RI_void = jnp.asarray(mat_dict[mat_pattern[0]])[:,jnp.newaxis,jnp.newaxis]
        self.delta_RI = (jnp.asarray(mat_dict[mat_pattern[1]]) - jnp.asarray(mat_dict[mat_pattern[0]]))[:,jnp.newaxis,jnp.newaxis]

        # Set Simulation Geometry
        self.Nx = Nx
        self.Ny = Ny
        self.thickness_pattern = thickness_pattern
        self.period = period
        self.primitive_lattice_vectors = basis.LatticeVectors(
            u=period[0] * basis.X,
            v=period[1] * basis.Y,
        )

        # Set Incidence Conditions
        self.in_plane_wavevector = jnp.asarray(in_plane_wavevector)
        self.diff_order = jnp.asarray(diff_order)

        # Solver Parameters
        self.truncation = truncation
        self.formulation = formulation

        # Cost Parameters
        self.IPR_exponent = IPR_exponent
    
    def set_accuracy(self, n_harmonic):
        self.basis_expansion = basis.generate_expansion(
            primitive_lattice_vectors=self.primitive_lattice_vectors,
            approximate_num_terms=n_harmonic,
            truncation=self.truncation,
        )

        self.coeffs = self.basis_expansion.basis_coefficients
        idx1 = jnp.where((self.coeffs[:,0] == self.diff_order[0,0]) & (self.coeffs[:,1] == self.diff_order[0,1]))[0][0]
        idx2 = jnp.where((self.coeffs[:,0] == self.diff_order[1,0]) & (self.coeffs[:,1] == self.diff_order[1,1]))[0][0]
        self.idx_tgt = jnp.array([idx1, idx2])

        # Compute incident power (normalization factor)
        density = jnp.zeros((self.Nx, self.Ny))
        permittivities = [
            self.eps_incident_medium,
            (self.delta_RI * density[jnp.newaxis,:,:] + self.RI_void)**2,
            self.eps_substrate,
        ]

        layer_solve_results = [
            fmm.eigensolve_isotropic_media(
                wavelength=self.lam,
                in_plane_wavevector=self.in_plane_wavevector,
                primitive_lattice_vectors=self.primitive_lattice_vectors,
                permittivity=p,
                expansion=self.basis_expansion,
                formulation=self.formulation,
            )
            for p in permittivities
        ]

        forward_amplitude_0_start = jnp.zeros((1, 2 * self.basis_expansion.num_terms, 2), dtype=complex)
        forward_amplitude_0_start = forward_amplitude_0_start.at[:,self.basis_expansion.num_terms,0].set(1)  # te
        forward_amplitude_0_start = forward_amplitude_0_start.at[:,0,1].set(1)  # tm
        incident_power, _ = fields.amplitude_poynting_flux(
            forward_amplitude=forward_amplitude_0_start,
            backward_amplitude=jnp.zeros_like(forward_amplitude_0_start),
            layer_solve_result=layer_solve_results[0],
        )
        self.incident_power = jnp.sum(incident_power, axis=1, keepdims=True)
        
        if hasattr(self, '_jitted_cost_fn'):
            del self._jitted_cost_fn
        if hasattr(self, '_jitted_grad_fn'):
            del self._jitted_grad_fn

    def get_cost(self, x, get_grad=False):
        if x.ndim == 1:
            density = jnp.asarray(x, dtype=float).reshape((self.Nx, self.Ny))
        else:
            density = jnp.asarray(x, dtype=float)

        cost_fn, grad_fn = self.jit_get_diffraction_costs

        if get_grad:
            cost, jac = grad_fn(density)
            jac = jax.device_get(jac)
            return cost, jac
            
        else:
            cost = cost_fn(density).block_until_ready()
            return cost

    @property
    def jit_get_diffraction_costs(self):
        # Create a closure or partial that captures 'self' parameters as static constants (where possible)
        # or just JIT the bound method if we are careful. 
        # Better: JIT a static function and pass self members as args.
        # Even better: Use a closure that JITs once per 'set_accuracy' (since basis expansion changes).
        
        # We need to re-jit only when set_accuracy changes self.basis_expansion and self.idx_tgt.
        # So we can cache the jitted function in set_accuracy or create it here on demand (cached).
        if not hasattr(self, '_jitted_cost_fn') or not hasattr(self, '_jitted_grad_fn'):
            self._create_jitted_cost_fn()
        return self._jitted_cost_fn, self._jitted_grad_fn

    def _create_jitted_cost_fn(self):
        # Define the function to be jitted, capturing current self state
        # We capture large arrays (eps_incident_medium, etc.) as captured variables in the closure.
        # JAX handles arrays in closures efficiently (as constants/traced values).
        
        # Local variables to capture (to ensure they are treated as constants/arrays by JIT)
        eps_incident = self.eps_incident_medium
        eps_substrate = self.eps_substrate
        RI_void = self.RI_void
        delta_RI = self.delta_RI
        thickness_pattern = self.thickness_pattern
        lam = self.lam
        in_plane_k = self.in_plane_wavevector
        lattice = self.primitive_lattice_vectors
        expansion = self.basis_expansion
        formulation = self.formulation
        idx_tgt = self.idx_tgt
        IPR_exponent = self.IPR_exponent
        incident_power = self.incident_power

        def _pure_cost_fn(density):
            permittivities = [
                eps_incident,
                (delta_RI * density[jnp.newaxis,:,:] + RI_void)**2,
                eps_substrate,
            ]
    
            thicknesses = [0.0, thickness_pattern, 0.0]
    
            layer_solve_results = [
                fmm.eigensolve_isotropic_media(
                    wavelength=lam,
                    in_plane_wavevector=in_plane_k,
                    primitive_lattice_vectors=lattice,
                    permittivity=p,
                    expansion=expansion,
                    formulation=formulation,
                )
                for p in permittivities
            ]
            
            s_matrix = scattering.stack_s_matrix(
                layer_solve_results=layer_solve_results,
                layer_thicknesses=[jnp.asarray(t) for t in thicknesses],
            )
    
            forward_amplitude_0_start = jnp.zeros((1, 2 * expansion.num_terms, 2), dtype=complex)
            forward_amplitude_0_start = forward_amplitude_0_start.at[:,expansion.num_terms,0].set(1)  # te
            forward_amplitude_0_start = forward_amplitude_0_start.at[:,0,1].set(1)  # tm

            forward_amplitude_N_start = s_matrix.s11 @ forward_amplitude_0_start
            transmitted_power, _ = fields.amplitude_poynting_flux(
                forward_amplitude=forward_amplitude_N_start,
                backward_amplitude=jnp.zeros_like(forward_amplitude_N_start),
                layer_solve_result=layer_solve_results[-1],
            )
            transmitted_power /= incident_power
            T_TE = transmitted_power[0, idx_tgt[0] + expansion.num_terms, 0]
            T_TM = transmitted_power[0, idx_tgt[1], 1]
    
            cost = -(T_TE**IPR_exponent + T_TM**IPR_exponent) / 2
            
            return cost
            
        self._jitted_cost_fn = jax.jit(_pure_cost_fn)
        self._jitted_grad_fn = jax.jit(jax.value_and_grad(_pure_cost_fn))
    
    def get_diffraction_and_fields(self, x):
        if x.ndim == 1:
            density = jnp.asarray(x, dtype=float).reshape((self.Nx, self.Ny))
        else:
            density = jnp.asarray(x, dtype=float)
        
        permittivities = [
            self.eps_incident_medium,
            (self.delta_RI * density[jnp.newaxis,:,:] + self.RI_void)**2,
            self.eps_substrate,
        ]

        layer_solve_results = [
            fmm.eigensolve_isotropic_media(
                wavelength=self.lam,
                in_plane_wavevector=self.in_plane_wavevector,
                primitive_lattice_vectors=self.primitive_lattice_vectors,
                permittivity=p,
                expansion=self.basis_expansion,
                formulation=self.formulation,
            )
            for p in permittivities
        ]

        # Diffraction Computation
        thicknesses = [0.0, self.thickness_pattern, 0.0]

        s_matrix = scattering.stack_s_matrix(
            layer_solve_results=layer_solve_results,
            layer_thicknesses=[jnp.asarray(t) for t in thicknesses],
        )

        forward_amplitude_0_start = jnp.zeros((1, 2 * self.basis_expansion.num_terms, 2), dtype=complex)
        forward_amplitude_0_start = forward_amplitude_0_start.at[:,self.basis_expansion.num_terms,0].set(1)  # te
        forward_amplitude_0_start = forward_amplitude_0_start.at[:,0,1].set(1)  # tm

        forward_amplitude_N_start = s_matrix.s11 @ forward_amplitude_0_start
        transmitted_power, _ = fields.amplitude_poynting_flux(
            forward_amplitude=forward_amplitude_N_start,
            backward_amplitude=jnp.zeros_like(forward_amplitude_N_start),
            layer_solve_result=layer_solve_results[-1],
        )
        transmitted_power /= self.incident_power
        T_TE = transmitted_power[0, self.idx_tgt[0] + self.basis_expansion.num_terms, 0]
        T_TM = transmitted_power[0, self.idx_tgt[1], 1]

        # Field Visualization
        thicknesses = [1.0, self.thickness_pattern, 4.0]

        s_matrices_interior = scattering.stack_s_matrices_interior(
            layer_solve_results=layer_solve_results,
            layer_thicknesses=thicknesses,
        )

        # Solve for the eigenmode amplitudes in every layer of the stack, given a
        # 0th order TE plane wave incident from the top.
        amplitudes_interior = fields.stack_amplitudes_interior(
            s_matrices_interior=s_matrices_interior,
            forward_amplitude_0_start=forward_amplitude_0_start,
            backward_amplitude_N_end=jnp.zeros_like(forward_amplitude_0_start),
        )

        # Compute the fields for a cross section at x = period[0] / 2
        y = jnp.linspace(0, self.period[1], self.Ny + 1)
        x = jnp.ones_like(y) * self.period[0] / 2
        (Ex, Ey, Ez), (Hx, Hy, Hz), (x, y, z) = fields.stack_fields_3d_on_coordinates(
            amplitudes_interior=amplitudes_interior,
            layer_solve_results=layer_solve_results,
            layer_thicknesses=thicknesses,
            layer_znum=[int(jnp.ceil(t / 0.01)) for t in thicknesses],
            x=x,
            y=y,
        )

        Ex = jax.device_get(Ex)
        Ey = jax.device_get(Ey)
        x = jax.device_get(x)
        y = jax.device_get(y)
        z = jax.device_get(z)

        return T_TE, T_TM, Ex, Ey, x, y, z, transmitted_power, self.incident_power, self.coeffs, self.idx_tgt