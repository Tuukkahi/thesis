import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.3)

fig, ax = plt.subplots(2,5)

def add_grid_lines(ax, x_num=5, y_num=5):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_grid = np.linspace(xlim[0], xlim[1], x_num)
    y_grid = np.linspace(ylim[0], ylim[1], y_num)
    for x in x_grid:
        ax.plot([x, x], [ylim[0], ylim[1]], '-', color='grey', alpha=0.2, linewidth=0.2)
    for y in y_grid:
        ax.plot([xlim[0], xlim[1]], [y, y], '-', color='grey', alpha=0.2, linewidth=0.2)

def plot_time_steps(actual, predict, flow_vectors, ax, fig):

  lags = actual.shape[0] - predict.shape[0]

  if flow_vectors is not None:
    assert predict.shape[0] == flow_vectors.shape[0]
    x_len = flow_vectors.shape[-2]
    y_len = flow_vectors.shape[-1]
    grid_x, grid_y = np.meshgrid(np.linspace(0, x_len - 1, x_len),
                                 np.linspace(0, y_len - 1, y_len))
    #grid_x /= x_len
    #grid_y /= y_len

  for i in range(actual.shape[0]):
    ax[1, i].set_box_aspect(1)
    ax[1,i].pcolormesh(actual[i], cmap='bone_r', vmin=0, vmax=1)
    #ax[0, i].contourf(grid_x, grid_y, actual[i], cmap='bone_r', extend='both')
    ax[1, i].set_yticks([])
    ax[1, i].set_xticks([])
    add_grid_lines(ax[0, i])
    if i == 0:
      ax[1, i].set_title("$y_{k}$")
    else:
      ax[1, i].set_title("$y_{k+" + str(i) + "}$")

  for i in range(predict.shape[0] + lags):
    if i < lags:
      ax[0, i].axis("off")
    else:
      ax[0, i].set_yticks([])
      ax[0, i].set_xticks([])
      ax[0, i].set_box_aspect(1)
      if i == 0:
          ax[0, i].set_title("$\overline{x}_{k \, \\vert \, k}$")
      else:
          ax[0, i].set_title("$\overline{x}_{k+" + str(i) + "\, \\vert \, k+" + str(i) + "}$")
      #ax[1, i].contourf(grid_x, grid_y, predict[i - lags], cmap='bone_r', extend='both')
      ax[0, i].pcolormesh(predict[i-lags], cmap='bone_r', vmin=0, vmax=1)
      add_grid_lines(ax[1, i])
      if flow_vectors is not None:
        ax[0, i].quiver(grid_x[::8,::8], grid_y[::8,::8], flow_vectors[i - lags, 0][::8,::8],
                        flow_vectors[i - lags, 1][::8,::8], alpha=0.3, linewidth=4)

  return fig

obs = np.load('experiments/fig/etkf_blocks_obs.npy')
xmean = np.load('experiments/fig/etkf_blocks_xmean.npy')
uv = np.load('experiments/fig/etkf_blocks_uvout.npy')
obs = obs[:,::2,::2]
xmean = xmean[:,::2,::2]
uv = uv[:,:,::2,::2]
obs[np.isnan(obs)] = 1e10
fig = plot_time_steps(obs, xmean, uv, ax, fig)

save()
