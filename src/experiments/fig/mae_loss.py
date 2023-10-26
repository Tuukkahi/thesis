import numpy as np
import matplotlib.pyplot as plt
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.35)

x = np.arange(15, 9*15, 15)
mae_loss_nn = [0.04525316, 0.06316551, 0.07602209, 0.08610436, 0.09437905, 0.10164249, 0.10805333, 0.11357881]
mae_loss_lk = [0.04282531, 0.05997251, 0.07237583, 0.08259787, 0.09119242, 0.09820219, 0.10454967, 0.11085155]
mae_loss_naive = [0.04743133, 0.06972325, 0.08400667, 0.09523758, 0.10472386, 0.112858, 0.1194752 , 0.12500473]

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
