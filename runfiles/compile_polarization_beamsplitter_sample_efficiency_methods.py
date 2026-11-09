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

print('\n\t*GEGD M0 CP0 ACV0', end='', flush=True)
cost_all_GEGD000 = np.zeros(10)
t_sim_GEGD000 = np.zeros(10)
convergence_iter_GEGD000 = np.zeros(10)
convergence_rate_GEGD000 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_M0CP0ACV0/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGD000[i] = j - N_convergence + 1
                t_sim_GEGD000[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGD000[i] = best_cost_hist[j]
                convergence_rate_GEGD000[i] = np.abs(best_cost_hist[0] - cost_all_GEGD000[i]) / t_sim_GEGD000[i]
                break

idx_best = np.argmin(cost_all_GEGD000)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD000[idx_best], flush=True)

print('\n\t*GEGD M1 CP0 ACV0', end='', flush=True)
cost_all_GEGD100 = np.zeros(10)
t_sim_GEGD100 = np.zeros(10)
convergence_iter_GEGD100 = np.zeros(10)
convergence_rate_GEGD100 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_M1CP0ACV0/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGD100[i] = j - N_convergence + 1
                t_sim_GEGD100[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGD100[i] = best_cost_hist[j]
                convergence_rate_GEGD100[i] = np.abs(best_cost_hist[0] - cost_all_GEGD100[i]) / t_sim_GEGD100[i]
                break

idx_best = np.argmin(cost_all_GEGD100)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD100[idx_best], flush=True)

print('\n\t*GEGD M0 CP1 ACV0', end='', flush=True)
cost_all_GEGD010 = np.zeros(10)
t_sim_GEGD010 = np.zeros(10)
convergence_iter_GEGD010 = np.zeros(10)
convergence_rate_GEGD010 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_M0CP1ACV0/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGD010[i] = j - N_convergence + 1
                t_sim_GEGD010[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGD010[i] = best_cost_hist[j]
                convergence_rate_GEGD010[i] = np.abs(best_cost_hist[0] - cost_all_GEGD010[i]) / t_sim_GEGD010[i]
                break

idx_best = np.argmin(cost_all_GEGD010)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD010[idx_best], flush=True)

print('\n\t*GEGD M0 CP0 ACV1', end='', flush=True)
cost_all_GEGD001 = np.zeros(10)
t_sim_GEGD001 = np.zeros(10)
convergence_iter_GEGD001 = np.zeros(10)
convergence_rate_GEGD001 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_M0CP0ACV1/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGD001[i] = j - N_convergence + 1
                t_sim_GEGD001[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGD001[i] = best_cost_hist[j]
                convergence_rate_GEGD001[i] = np.abs(best_cost_hist[0] - cost_all_GEGD001[i]) / t_sim_GEGD001[i]
                break

idx_best = np.argmin(cost_all_GEGD001)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD001[idx_best], flush=True)

print('\n\t*GEGD M0 CP1 ACV1', end='', flush=True)
cost_all_GEGD011 = np.zeros(10)
t_sim_GEGD011 = np.zeros(10)
convergence_iter_GEGD011 = np.zeros(10)
convergence_rate_GEGD011 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_M0CP1ACV1/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGD011[i] = j - N_convergence + 1
                t_sim_GEGD011[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGD011[i] = best_cost_hist[j]
                convergence_rate_GEGD011[i] = np.abs(best_cost_hist[0] - cost_all_GEGD011[i]) / t_sim_GEGD011[i]
                break

idx_best = np.argmin(cost_all_GEGD011)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD011[idx_best], flush=True)

print('\n\t*GEGD M1 CP0 ACV1', end='', flush=True)
cost_all_GEGD101 = np.zeros(10)
t_sim_GEGD101 = np.zeros(10)
convergence_iter_GEGD101 = np.zeros(10)
convergence_rate_GEGD101 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_M1CP0ACV1/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGD101[i] = j - N_convergence + 1
                t_sim_GEGD101[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGD101[i] = best_cost_hist[j]
                convergence_rate_GEGD101[i] = np.abs(best_cost_hist[0] - cost_all_GEGD101[i]) / t_sim_GEGD101[i]
                break

idx_best = np.argmin(cost_all_GEGD101)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD101[idx_best], flush=True)

print('\n\t*GEGD M1 CP1 ACV0', end='', flush=True)
cost_all_GEGD110 = np.zeros(10)
t_sim_GEGD110 = np.zeros(10)
convergence_iter_GEGD110 = np.zeros(10)
convergence_rate_GEGD110 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD_M1CP1ACV0/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGD110[i] = j - N_convergence + 1
                t_sim_GEGD110[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGD110[i] = best_cost_hist[j]
                convergence_rate_GEGD110[i] = np.abs(best_cost_hist[0] - cost_all_GEGD110[i]) / t_sim_GEGD110[i]
                break

idx_best = np.argmin(cost_all_GEGD110)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD110[idx_best], flush=True)

print('\n\t*GEGD M1 CP1 ACV1', end='', flush=True)
cost_all_GEGD111 = np.zeros(10)
t_sim_GEGD111 = np.zeros(10)
convergence_iter_GEGD111 = np.zeros(10)
convergence_rate_GEGD111 = np.zeros(10)
for i in range(10):
    with np.load(directory + data_dir + "/GEGD/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_sig_ens0.01_eta5e-05_mfs7_exp20_try" + str(i + 1) + "_GEGD_results.npz") as data:
        best_cost_hist = data['best_cost_hist']
        N_high_fidelity_hist = data['N_high_fidelity_hist']
        N_low_fidelity_hist = data['N_low_fidelity_hist']
        for j in range(best_cost_hist.size):
            if j >= N_convergence and np.all(best_cost_hist[j-N_convergence+1:j] <= best_cost_hist[j] + 1e-3):
                convergence_iter_GEGD111[i] = j - N_convergence + 1
                t_sim_GEGD111[i] = t_high_fidelity * np.sum(N_high_fidelity_hist[:j-N_convergence+1]) + t_low_fidelity * np.sum(N_low_fidelity_hist[:j-N_convergence+1])
                cost_all_GEGD111[i] = best_cost_hist[j]
                convergence_rate_GEGD111[i] = np.abs(best_cost_hist[0] - cost_all_GEGD111[i]) / t_sim_GEGD111[i]
                break

idx_best = np.argmin(cost_all_GEGD111)
print(' --> Best Cost (idx=',idx_best+1,'): ',cost_all_GEGD111[idx_best], flush=True)

np.savez(directory + data_dir + '/polarization_beamsplitter_IPR1_Nensemble20_Ndim45x90_D1_mfs7_sample_efficiency_methods',
    cost_all_GEGD000=cost_all_GEGD000,
    cost_all_GEGD100=cost_all_GEGD100,
    cost_all_GEGD010=cost_all_GEGD010,
    cost_all_GEGD001=cost_all_GEGD001,
    cost_all_GEGD011=cost_all_GEGD011,
    cost_all_GEGD101=cost_all_GEGD101,
    cost_all_GEGD110=cost_all_GEGD110,
    cost_all_GEGD111=cost_all_GEGD111,
    t_sim_GEGD000=t_sim_GEGD000,
    t_sim_GEGD100=t_sim_GEGD100,
    t_sim_GEGD010=t_sim_GEGD010,
    t_sim_GEGD001=t_sim_GEGD001,
    t_sim_GEGD011=t_sim_GEGD011,
    t_sim_GEGD101=t_sim_GEGD101,
    t_sim_GEGD110=t_sim_GEGD110,
    t_sim_GEGD111=t_sim_GEGD111,
    convergence_rate_GEGD000=convergence_rate_GEGD000,
    convergence_rate_GEGD100=convergence_rate_GEGD100,
    convergence_rate_GEGD010=convergence_rate_GEGD010,
    convergence_rate_GEGD001=convergence_rate_GEGD001,
    convergence_rate_GEGD011=convergence_rate_GEGD011,
    convergence_rate_GEGD101=convergence_rate_GEGD101,
    convergence_rate_GEGD110=convergence_rate_GEGD110,
    convergence_rate_GEGD111=convergence_rate_GEGD111,
)

t2 = time.time()
print('>>> Time taken: ' + str(np.round(t2 - t1, 2)) + ' s', flush=True)