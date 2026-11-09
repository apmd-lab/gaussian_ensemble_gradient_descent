import os
directory = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('/home/fs01/sm3266/gaussian_ensemble_gradient_descent')

import numpy as np
import time

# Geometry
Nx = 45
Ny = 90
symmetry = 1 # Currently supported: (None), (D1,2,4)
periodic = 1
padding = None
min_feature_size = 7 # minimum feature size in pixels
d_pixel = 0.01 # pixel side length (nm)
feasible_design_generation_method = 'brush' # brush / two_phase_projection

t_low_fidelity = 0.18 # low-fidelity simulation time in seconds
t_high_fidelity = 1.45 # high-fidelity simulation time in seconds
t_fwd_AD = 1.61
N_convergence = 100

data_dir = "/RCWA_functions/polarization_beamsplitter/run_until_convergence"

t1 = time.time()
print('### Running Simulations', flush=True)

print('\n\t*GEGD reference', end='', flush=True)
cost_all_GEGD = np.zeros(10)
t_sim_GEGD = np.zeros(10)
convergence_iter_GEGD = np.zeros(10)
convergence_rate_GEGD = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGD[i] = j - N_convergence + 1
                t_sim_GEGD[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGD[i] = best_cost_hist[j]
                convergence_rate_GEGD[i] = np.abs(best_cost_hist[0] - cost_all_GEGD[i]) / t_sim_GEGD[i]
                break

idx_best = np.argmin(cost_all_GEGD)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD[idx_best], flush=True)

print('\n\t*GEGD eta5e-4', end='', flush=True)
cost_all_GEGDeta4 = np.zeros(10)
t_sim_GEGDeta4 = np.zeros(10)
convergence_iter_GEGDeta4 = np.zeros(10)
convergence_rate_GEGDeta4 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_eta5e-4/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta0.0005_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGDeta4[i] = j - N_convergence + 1
                t_sim_GEGDeta4[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGDeta4[i] = best_cost_hist[j]
                convergence_rate_GEGDeta4[i] = np.abs(best_cost_hist[0] - cost_all_GEGDeta4[i]) / t_sim_GEGDeta4[i]
                break

idx_best = np.argmin(cost_all_GEGDeta4)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGDeta4[idx_best], flush=True)

print('\n\t*GEGD eta6', end='', flush=True)
cost_all_GEGDeta6 = np.zeros(10)
t_sim_GEGDeta6 = np.zeros(10)
convergence_iter_GEGDeta6 = np.zeros(10)
convergence_rate_GEGDeta6 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_eta5e-6/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-06_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGDeta6[i] = j - N_convergence + 1
                t_sim_GEGDeta6[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGDeta6[i] = best_cost_hist[j]
                convergence_rate_GEGDeta6[i] = np.abs(best_cost_hist[0] - cost_all_GEGDeta6[i]) / t_sim_GEGDeta6[i]
                break

idx_best = np.argmin(cost_all_GEGDeta6)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGDeta6[idx_best], flush=True)

print('\n\t*GEGD Nensemble10', end='', flush=True)
cost_all_GEGDNensemble10 = np.zeros(10)
t_sim_GEGDNensemble10 = np.zeros(10)
convergence_iter_GEGDNensemble10 = np.zeros(10)
convergence_rate_GEGDNensemble10 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_Nensemble10/polarization_beamsplitter_IPR1_Nensemble10_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGDNensemble10[i] = j - N_convergence + 1
                t_sim_GEGDNensemble10[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGDNensemble10[i] = best_cost_hist[j]
                convergence_rate_GEGDNensemble10[i] = np.abs(best_cost_hist[0] - cost_all_GEGDNensemble10[i]) / t_sim_GEGDNensemble10[i]
                break

idx_best = np.argmin(cost_all_GEGDNensemble10)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGDNensemble10[idx_best], flush=True)

print('\n\t*GEGD Nensemble40', end='', flush=True)
cost_all_GEGDNensemble40 = np.zeros(10)
t_sim_GEGDNensemble40 = np.zeros(10)
convergence_iter_GEGDNensemble40 = np.zeros(10)
convergence_rate_GEGDNensemble40 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_Nensemble40/polarization_beamsplitter_IPR1_Nensemble40_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGDNensemble40[i] = j - N_convergence + 1
                t_sim_GEGDNensemble40[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGDNensemble40[i] = best_cost_hist[j]
                convergence_rate_GEGDNensemble40[i] = np.abs(best_cost_hist[0] - cost_all_GEGDNensemble40[i]) / t_sim_GEGDNensemble40[i]
                break

idx_best = np.argmin(cost_all_GEGDNensemble40)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGDNensemble40[idx_best], flush=True)

print('\n\t*GEGD exp10', end='', flush=True)
cost_all_GEGDexp10 = np.zeros(10)
t_sim_GEGDexp10 = np.zeros(10)
convergence_iter_GEGDexp10 = np.zeros(10)
convergence_rate_GEGDexp10 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_coeff_exp10/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp10_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGDexp10[i] = j - N_convergence + 1
                t_sim_GEGDexp10[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGDexp10[i] = best_cost_hist[j]
                convergence_rate_GEGDexp10[i] = np.abs(best_cost_hist[0] - cost_all_GEGDexp10[i]) / t_sim_GEGDexp10[i]
                break

idx_best = np.argmin(cost_all_GEGDexp10)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGDexp10[idx_best], flush=True)

print('\n\t*GEGD exp30', end='', flush=True)
cost_all_GEGDexp30 = np.zeros(10)
t_sim_GEGDexp30 = np.zeros(10)
convergence_iter_GEGDexp30 = np.zeros(10)
convergence_rate_GEGDexp30 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_coeff_exp30/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp30_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGDexp30[i] = j - N_convergence + 1
                t_sim_GEGDexp30[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGDexp30[i] = best_cost_hist[j]
                convergence_rate_GEGDexp30[i] = np.abs(best_cost_hist[0] - cost_all_GEGDexp30[i]) / t_sim_GEGDexp30[i]
                break

idx_best = np.argmin(cost_all_GEGDexp30)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGDexp30[idx_best], flush=True)

np.savez(directory + data_dir + '/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_mfs7_hyperparameters',
    cost_all_GEGD=cost_all_GEGD,
    cost_all_GEGDeta4=cost_all_GEGDeta4,
    cost_all_GEGDeta6=cost_all_GEGDeta6,
    cost_all_GEGDNensemble10=cost_all_GEGDNensemble10,
    cost_all_GEGDNensemble40=cost_all_GEGDNensemble40,
    cost_all_GEGDexp10=cost_all_GEGDexp10,
    cost_all_GEGDexp30=cost_all_GEGDexp30,
    t_sim_GEGD=t_sim_GEGD,
    t_sim_GEGDeta4=t_sim_GEGDeta4,
    t_sim_GEGDeta6=t_sim_GEGDeta6,
    t_sim_GEGDNensemble10=t_sim_GEGDNensemble10,
    t_sim_GEGDNensemble40=t_sim_GEGDNensemble40,
    t_sim_GEGDexp10=t_sim_GEGDexp10,
    t_sim_GEGDexp30=t_sim_GEGDexp30,
    convergence_rate_GEGD=convergence_rate_GEGD,
    convergence_rate_GEGDeta4=convergence_rate_GEGDeta4,
    convergence_rate_GEGDeta6=convergence_rate_GEGDeta6,
    convergence_rate_GEGDNensemble10=convergence_rate_GEGDNensemble10,
    convergence_rate_GEGDNensemble40=convergence_rate_GEGDNensemble40,
    convergence_rate_GEGDexp10=convergence_rate_GEGDexp10,
    convergence_rate_GEGDexp30=convergence_rate_GEGDexp30,
)

t2 = time.time()
print('>>> Time taken: ' + str(np.round(t2 - t1, 2)) + ' s', flush=True)