import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
#sys.path.append('/home/minseokhwan/gaussian_ensemble_gradient_descent')
sys.path.append('/home/fs01/sm3266/gaussian_ensemble_gradient_descent')

import gc
import jax
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import time

Nthreads = 20
# Let SLURM handle GPU assignment via GRES — don't override CUDA_VISIBLE_DEVICES
# cuda_ind = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_ind)

# Geometry
Nx = 100
Ny = 100
symmetry = 3 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
min_feature_size = 7 # minimum feature size in pixels
d_pixel = 0.01 # pixel side length (nm)
feasible_design_generation_method = 'brush' # brush / two_phase_projection
upsampling_ratio = 9

# Define Cost Object
#--------------------------------------------------------------------------
# class "custom_objective" must have the following methods: 
# (1) get_cost(x, get_grad) --> return cost & gradient(if get_grad==True)
# (2) set_accuracy(setting)
#--------------------------------------------------------------------------
import RCWA_functions.RGB_color_router_FMMAX as objfun

IPR_exponent = 1/1

lam = np.linspace(0.400, 0.700, 301) #np.array([0.650,0.550,0.450]) # um
theta_inc = np.array([0])*np.pi/180
phi_inc = np.array([0])*np.pi/180
in_plane_wavevector = np.array([0.0, 0.0])

period = np.array([Nx * d_pixel, Ny * d_pixel])
thickness_background = 1.0
thickness_pattern = 0.7
thickness_substrate = 1.0

mat_background = np.array(['Air']) # background (incident side)
mat_pattern = np.array(['Air','Si3N4_Luke']) # Low RI, High RI
mat_substrate = np.array(['Si_Schinke_Shkondin'])

# Optimizer Settings
#--------------------------------------------------------------------------------------------------------------------------
# Run a convergence test to determine the settings for the low and high-fidelity simulations
# high-fidelity: accuracy required for actual application
# low-fidelity: faster and less accurate, but accurate enough to ensure high correlation with the high-fidelity simulations
#--------------------------------------------------------------------------------------------------------------------------
low_fidelity_setting = 18**2 # low-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)
high_fidelity_setting = 40**2 # high-fidelity simulation setting (e.g. RCWA: number of harmonics, FDTD: mesh density, etc.)

# Process wavelengths in chunks to limit GPU memory usage
lam_chunk_size = 5

def simulate_chunked(design, upsampling_ratio_arg):
    """Run get_diffraction_and_fields over lam in chunks and concatenate results."""
    n_chunks = int(np.ceil(len(lam) / lam_chunk_size))
    flux_chunks = []
    R_flux_chunks = []
    G_flux_chunks = []
    B_flux_chunks = []
    incident_flux_chunks = []

    for ic in range(n_chunks):
        lam_chunk = lam[ic * lam_chunk_size : (ic + 1) * lam_chunk_size]
        print(f'\n\t\tChunk {ic+1}/{n_chunks} ({len(lam_chunk)} wavelengths)', end='', flush=True)

        obj = objfun.custom_objective(
            Nx,
            Ny,
            period,
            thickness_background,
            thickness_pattern,
            thickness_substrate,
            lam_chunk,
            in_plane_wavevector,
            mat_background,
            mat_pattern,
            mat_substrate,
            IPR_exponent=1/1,
            precision='float64',
        )
        obj.set_accuracy(high_fidelity_setting)

        f, Rf, Gf, Bf, inc_f = obj.get_diffraction_and_fields(design, upsampling_ratio_arg)
        flux_chunks.append(np.asarray(f))
        R_flux_chunks.append(np.asarray(Rf))
        G_flux_chunks.append(np.asarray(Gf))
        B_flux_chunks.append(np.asarray(Bf))
        incident_flux_chunks.append(np.asarray(inc_f))

        # Free GPU memory before next chunk
        del obj, f, Rf, Gf, Bf, inc_f
        jax.clear_caches()
        gc.collect()

    return (
        np.concatenate(flux_chunks, axis=0),
        np.concatenate(R_flux_chunks, axis=0),
        np.concatenate(G_flux_chunks, axis=0),
        np.concatenate(B_flux_chunks, axis=0),
        np.concatenate(incident_flux_chunks, axis=0),
    )

t1 = time.time()
print('### Running Simulations', flush=True)

print('\n\t*GEGD', end='', flush=True)
cost_all_GEGD = np.zeros(10)
for i in range(10):
    with np.load(directory + "/RCWA_functions/RGB_color_router/GEGD/RGB_color_router_IPR1_Nensemble20_Ndim100x100_D3_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        cost_all_GEGD[i] = data['best_cost_hist'][-1]

idx_best = np.argmin(cost_all_GEGD)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD[idx_best], flush=True)

with np.load(directory + "/RCWA_functions/RGB_color_router/GEGD/RGB_color_router_IPR1_Nensemble20_Ndim100x100_D3_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(idx_best + 1) + "_GEGD_results.npz") as data:
    x = data['best_x_final'].reshape(Nx, Ny)
'''
print('\n\t  Simulating native resolution...', flush=True)
flux, R_flux, G_flux, B_flux, incident_flux = simulate_chunked(x, 1)
x_up = np.where(gaussian_filter(zoom(x.astype(np.float64), upsampling_ratio, order=1, mode='wrap'), sigma=upsampling_ratio, mode='wrap') > 0.5, 1, 0)
print('\n\t  Simulating upsampled resolution...', flush=True)
flux_up, R_flux_up, G_flux_up, B_flux_up, incident_flux_up = simulate_chunked(x_up, upsampling_ratio)
'''
print('\n\t*TF-BFGS', end='', flush=True)
cost_all_BFGS = np.zeros(108)
cost_all_BFGS_mfs = np.zeros(108)
for i in range(6):
    with np.load(directory + "/RCWA_functions/RGB_color_router/TF_BFGS/RGB_color_router_IPR1_Ntrial18_Ndim100x100_D3_mfs7_try" + str(i + 1) + "_TF_results.npz") as data:
        cost_all_BFGS[18*i:18*(i+1)] = data['cost_fin'][0,:]
        cost_all_BFGS_mfs[18*i:18*(i+1)] = data['cost_fin'][1,:]

idx_best = np.argmin(cost_all_BFGS)
idx_best_mfs = np.argmin(cost_all_BFGS_mfs)
print(' --> Best Cost (idx=',idx_best+1,', idx_best_mfs=',idx_best_mfs,'): ',cost_all_BFGS[idx_best],' / ',cost_all_BFGS_mfs[idx_best_mfs],' (mfs not enforced / enforced)', flush=True)

print('\n\t*sep-CMA-ES', end='', flush=True)
cost_all_sep_CMA_ES = np.zeros(10)
for i in range(10):
    with np.load(directory + "/RCWA_functions/RGB_color_router/sep_CMA_ES/RGB_color_router_IPR1_Nensemble20_Ndim100x100_D3_mfs7_try" + str(i + 1) + "_sep_CMA_ES_results.npz") as data:
        cost_all_sep_CMA_ES[i] = data['best_cost_hist'][-1]

idx_best = np.argmin(cost_all_sep_CMA_ES)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_sep_CMA_ES[idx_best], flush=True)

print('\n\t*GA', end='', flush=True)
cost_all_GA = np.zeros(10)
for i in range(10):
    with np.load(directory + "/RCWA_functions/RGB_color_router/GA/RGB_color_router_IPR1_Nensemble20_Ndim100x100_D3_mfs7_try" + str(i + 1) + "_AF_GA_results.npz") as data:
        cost_all_GA[i] = data['best_cost_hist'][-1]

idx_best = np.argmin(cost_all_GA)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GA[idx_best], flush=True)

print('\n\t*AF-STE', end='', flush=True)
cost_all_AF_STE = np.zeros(144)
for i in range(8):
    with np.load(directory + "/RCWA_functions/RGB_color_router/AF_STE/RGB_color_router_IPR1_Ntrial18_Ndim100x100_D3_eta0.01_mfs7_try" + str(i + 1) + "_AF_STE_results.npz") as data:
        cost_all_AF_STE[18*i:18*(i+1)] = np.min(data['cost_hist'], axis=0)

idx_best = np.argmin(cost_all_AF_STE)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_AF_STE[idx_best], flush=True)

np.savez(directory + '/RCWA_functions/RGB_color_router/RGB_color_router_IPR1_Nensemble20_Ndim100x100_D3_mfs7_simulations',
    cost_all_GEGD=cost_all_GEGD,
    cost_all_BFGS=cost_all_BFGS,
    cost_all_BFGS_mfs=cost_all_BFGS_mfs,
    cost_all_sep_CMA_ES=cost_all_sep_CMA_ES,
    cost_all_GA=cost_all_GA,
    cost_all_AF_STE=cost_all_AF_STE,
    lam=lam,
    x=x,
)
'''
    x_up=x_up,
    flux=flux,
    R_flux=R_flux,
    G_flux=G_flux,
    B_flux=B_flux,
    incident_flux=incident_flux,
    flux_up=flux_up,
    R_flux_up=R_flux_up,
    G_flux_up=G_flux_up,
    B_flux_up=B_flux_up,
    incident_flux_up=incident_flux_up,
)
'''

t2 = time.time()
print('>>> Time taken: ' + str(np.round(t2 - t1, 2)) + ' s', flush=True)