import numpy as np
import matplotlib.pyplot as plt
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.35)

x = np.arange(15, 9*15, 15)
mae_loss_nn = [0.04593692, 0.06405915, 0.07670248, 0.08639832, 0.09433789, 0.10203682, 0.10822865, 0.11336926]
mae_loss_lk = [0.04232488, 0.05965823, 0.07221683, 0.0826459 , 0.09126299, 0.09853887, 0.10513083, 0.11177037]
mae_loss_naive = [0.04657366, 0.06860248, 0.08302534, 0.09415203, 0.10359123, 0.11202101, 0.11895723, 0.12509705]

plt.plot(x, mae_loss_nn, 'k:', marker='s',fillstyle='none', label='Neural Network')
plt.plot(x, mae_loss_lk, 'k:', marker='v', fillstyle='none', label='Lucas-Kanade')
plt.plot(x, mae_loss_naive, 'k:', marker='d', fillstyle='none', label='Naive')

plt.xticks(x)
plt.ylim(0.035,0.13)
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.ylabel('MAE')
plt.xlabel('Prediction Lead Time (min)')
plt.grid(color='grey', linewidth=1, linestyle='-', alpha=0.2)


save()
