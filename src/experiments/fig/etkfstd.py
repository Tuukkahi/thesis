import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.2)

fig, ax = plt.subplots(1, 6, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.18]})
ax[-1].axis("off")

def add_grid_lines(ax, x_num=5, y_num=5):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_grid = np.linspace(xlim[0], xlim[1], x_num)
    y_grid = np.linspace(ylim[0], ylim[1], y_num)
    for x in x_grid:
        ax.plot([x, x], [ylim[0], ylim[1]], '-', color='grey', alpha=0.2, linewidth=0.2)
    for y in y_grid:
        ax.plot([xlim[0], xlim[1]], [y, y], '-', color='grey', alpha=0.2, linewidth=0.2)

def plot_time_steps(data, ax, fig):

  vmin = np.min(data)
  vmax = np.min(data)

  for i in range(data.shape[0]):
    ax[i].set_box_aspect(1)
    im = ax[i].pcolormesh(data[i], cmap='bone_r', vmin=0, vmax=1)
    #ax[0, i].contourf(grid_x, grid_y, data[i], cmap='bone_r', extend='both')
    ax[i].set_yticks([])
    ax[i].set_xticks([])
    add_grid_lines(ax[i])
    ax[i].set_title('t+' + str(i*15) + ' min')

  cbar_ax = fig.add_axes([0.94, 0.2, 0.01, 0.6])
  plt.colorbar(im, cax=cbar_ax)
  return fig, ax,

xout_std = np.load('experiments/fig/etkf_cot_mean_out_std_NN.npy')
xout_std = xout_std[:5,::3,::3]
fig, ax = plot_time_steps(2*xout_std, ax, fig)

save()
