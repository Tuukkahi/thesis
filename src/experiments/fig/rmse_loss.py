import numpy as np
import matplotlib.pyplot as plt
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.35)

x = np.arange(15, 9*15, 15)
mse_loss_nn = np.array([0.00614594, 0.01074895, 0.01459483, 0.01788927, 0.02085317, 0.02402384, 0.02683662, 0.02947479])
mse_loss_lk = np.array([0.00517818, 0.00926325, 0.0128694 , 0.01633907, 0.01954229, 0.02243342, 0.02533811, 0.02842341])
mse_loss_naive = np.array([0.00635762, 0.01219122, 0.01666103, 0.0206661 , 0.02452978, 0.02817327, 0.03155731, 0.03487181])

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
