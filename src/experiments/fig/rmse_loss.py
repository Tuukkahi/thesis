import numpy as np
import matplotlib.pyplot as plt
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.35)

x = np.arange(15, 9*15, 15)
mse_loss_nn = np.array([0.00609066, 0.01072457, 0.0146978 , 0.01818584, 0.02135012, 0.02429144, 0.02716422, 0.02979652])
mse_loss_lk = np.array([0.0054561 , 0.00959998, 0.01325985, 0.01670621, 0.01983971, 0.02258265, 0.02533306, 0.02820123])
mse_loss_naive = np.array([0.00679733, 0.01290859, 0.01739062, 0.0215012 , 0.02537579, 0.02885803, 0.0320196 , 0.03482411])

plt.plot(x, np.sqrt(mse_loss_nn), 'k:', marker='s',fillstyle='none', label='Neural Network')
plt.plot(x, np.sqrt(mse_loss_lk), 'k:', marker='v', fillstyle='none', label='Lucas-Kanade')
plt.plot(x, np.sqrt(mse_loss_naive), 'k:', marker='d', fillstyle='none', label='Naive')

plt.xticks(x)
plt.ylim(0.065,0.195)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.ylabel('RMSE')
plt.xlabel('Prediction Lead Time (min)')
plt.grid(color='grey', linewidth=1, linestyle='-', alpha=0.2)


save()
