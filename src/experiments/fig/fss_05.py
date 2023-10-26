import numpy as np
import matplotlib.pyplot as plt
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.35)

fig, ax = plt.subplots(2,2,gridspec_kw={'height_ratios': [20, 1]})
x = np.arange(15, 9*15, 15)
fss_nn = np.array([0.46804624, 0.2647375 , 0.17591129, 0.12824737, 0.09450144, 0.06486478, 0.03705563, 0.02498489])
fss_lk = np.array([0.49860354, 0.32033379, 0.22572263, 0.16304816, 0.12662241, 0.08724261, 0.05112378, 0.03949873])
fss_naive = np.array([0.40156848, 0.20624497, 0.14000424, 0.09172761, 0.06104271, 0.05165397, 0.0321028 , 0.03078491])

ax[0,0].plot(x, fss_nn, 'k:', marker='s',fillstyle='none', label='Neural Network')
ax[0,0].plot(x, fss_lk, 'k:', marker='v', fillstyle='none', label='Lucas-Kanade')
ax[0,0].plot(x, fss_naive, 'k:', marker='d', fillstyle='none', label='Naive')

ax[0,0].set_xticks(x)
ax[0,0].set_ylim(0.0,1.0)
ax[0,0].set_ylabel('FSS (thr=0.7, window size=8)')
ax[0,0].set_xlabel('Prediction Lead Time (min)')
ax[0,0].grid(color='grey', linewidth=1, linestyle='-', alpha=0.2)

fss_nn = np.array([0.65101815, 0.50983101, 0.41177463, 0.34508235, 0.29527532, 0.24607908, 0.19832885, 0.16743786])
fss_lk = np.array([0.67250362, 0.5350313 , 0.43301533, 0.37107398, 0.32030087, 0.26843706, 0.22105356, 0.18390378])
fss_naive = np.array([0.61062017, 0.45353944, 0.35554105, 0.29915689, 0.26337637, 0.22487005, 0.18943395, 0.16595471])

ax[0,1].plot(x, fss_nn, 'k:', marker='s',fillstyle='none', label='Neural Network')
ax[0,1].plot(x, fss_lk, 'k:', marker='v', fillstyle='none', label='Lucas-Kanade')
ax[0,1].plot(x, fss_naive, 'k:', marker='d', fillstyle='none', label='Naive')

ax[0,1].set_xticks(x)
ax[0,1].set_ylim(0.0,1.0)
ax[0,1].set_ylabel('FSS (thr=0.4, window size=5)')
ax[0,1].set_xlabel('Prediction Lead Time (min)')
ax[0,1].grid(color='grey', linewidth=1, linestyle='-', alpha=0.2)

ax[1][0].axis("off")
ax[1][1].axis("off")

handles, labels = ax[0,0].get_legend_handles_labels()

fig.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3)
save(fig)
