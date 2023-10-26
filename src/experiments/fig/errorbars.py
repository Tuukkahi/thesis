import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.35)

fig, ax = plt.subplots()


xout_std = np.load('experiments/fig/etkf_cot_mean_out_std_NN.npy')[:5]
xprior_std_mean = np.load('experiments/fig/etkf_cot_mean_prior_std_mean_NN.npy')[:5]
xprior_mean = np.load('experiments/fig/etkf_cot_mean_prior_NN.npy')[:5]
obs = np.load('experiments/fig/etkf_cot_obs.npy')[:5]
xout_mean = np.load('experiments/fig/etkf_cot_mean_out_NN.npy')[:5]

xout_std_mean = xout_std.mean(axis=(1,2))
xout_mean[~np.isfinite(obs)] = np.nan
xprior_mean[~np.isfinite(obs)] = np.nan
time = np.arange(xout_std_mean.shape[0])

trans1 = Affine2D().translate(-0.15, 0.0) + ax.transData
trans2 = Affine2D().translate(+0.15, 0.0) + ax.transData

eb1 = plt.errorbar(time, np.nanmean(xprior_mean,axis=(1,2)), 2*xprior_std_mean, label='Prior', marker="o", linestyle="none", transform=trans1, color='black', fillstyle='none')
eb1[-1][0].set_linestyle(':')
eb2 = plt.errorbar(time, np.nanmean(obs, axis=(1,2)), 0.233, marker='s', label='Observation', linestyle="none", color='black', fillstyle='none')
eb2[-1][0].set_linestyle('-.')
eb3 = plt.errorbar(time, np.nanmean(xout_mean,axis=(1,2)), 2*xout_std_mean, marker='s', label='Posterior', linestyle="none", transform=trans2, color='black', fillstyle='none')
eb3[-1][0].set_linestyle('--')

ax.set_ylabel('COT image mean')
ax.set_xlabel('t')
plt.grid(color='grey', linewidth=1, linestyle='-', alpha=0.2)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

save()
