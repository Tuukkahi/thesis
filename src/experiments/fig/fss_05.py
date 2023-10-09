import numpy as np
import matplotlib.pyplot as plt
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.35)

x = np.arange(15, 9*15, 15)
fss_nn = np.array([0.56115314, 0.4012244 , 0.29710576, 0.23280209, 0.18682052, 0.13439911, 0.10098008, 0.08409548])
fss_lk = np.array([0.59859308, 0.44802801, 0.34638292, 0.27774223, 0.2210131, 0.16335104, 0.12053218, 0.09616703])
fss_naive = np.array([0.51722272, 0.33989148, 0.25247397, 0.19729096, 0.15848955, 0.12655331, 0.09785977, 0.0852008])

plt.plot(x, fss_nn, 'k:', marker='s',fillstyle='none', label='Neural Network')
plt.plot(x, fss_lk, 'k:', marker='v', fillstyle='none', label='Lucas-Kanade')
plt.plot(x, fss_naive, 'k:', marker='d', fillstyle='none', label='Naive')

plt.xticks(x)
plt.ylim(0.0,1.0)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.ylabel('FSS (thr=0.5)')
plt.xlabel('Prediction Lead Time (min)')
plt.grid(color='grey', linewidth=1, linestyle='-', alpha=0.2)


save()
