import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.9)

fig, ax = plt.subplots(5,4)

def add_grid_lines(ax, x_num=5, y_num=5):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_grid = np.linspace(xlim[0], xlim[1], x_num)
    y_grid = np.linspace(ylim[0], ylim[1], y_num)
    for x in x_grid:
        ax.plot([x, x], [ylim[0], ylim[1]], '-', color='grey', alpha=0.2, linewidth=0.2)
    for y in y_grid:
        ax.plot([xlim[0], xlim[1]], [y, y], '-', color='grey', alpha=0.2, linewidth=0.2)

def plot_time_steps(obs, predict, flow_vectors, ax, fig):

  if flow_vectors is not None:
    x_len = flow_vectors[0].shape[-2]
    y_len = flow_vectors[0].shape[-1]
    grid_x, grid_y = np.meshgrid(np.linspace(0, x_len - 1, x_len),
                                 np.linspace(0, y_len - 1, y_len))
    #grid_x /= x_len
    #grid_y /= y_len
    
  for i in range(obs.shape[0]):
    ax[i, 0].set_box_aspect(1)
    ax[i, 0].pcolormesh(obs[i], cmap='bone_r', vmin=-0.1, vmax=8)
    #ax[0, i].contourf(grid_x, grid_y, actual[i], cmap='bone_r', extend='both')
    ax[i, 0].set_yticks([])
    ax[i, 0].set_xticks([])
    ax[i,0].set_ylabel('t+' + str(i*15) + 'min')
    add_grid_lines(ax[i, 0])
    for j in range(len(predict)):
        ax[i, j+1].set_box_aspect(1)
        ax[i, j+1].pcolormesh(predict[j][i], cmap='bone_r', vmin=-0.1, vmax=8)
        ax[i, j+1].set_yticks([])
        ax[i, j+1].set_xticks([])
        add_grid_lines(ax[i, j+1])
        ax[i, j+1].quiver(grid_x[::8,::8], grid_y[::8,::8], flow_vectors[j][i, 0][::8,::8],
                        flow_vectors[j][i, 1][::8,::8], alpha=0.3, linewidth=4)

  ax[0, 0].set_title("Observation")
  ax[0, 1].set_title("NN")
  ax[0, 2].set_title("LK")
  ax[0, 3].set_title("I")
  return fig

obs = np.load('experiments/fig/etkf_cot_obs.npy')[:5,::2,::2]
obs[~np.isfinite(obs)] = 15

uv_NN = np.load('experiments/fig/etkf_cot_uv_NN.npy')[:5,:,::2,::2]
uv_LK = np.load('experiments/fig/etkf_cot_uv_LK.npy')[:5,:,::2,::2]
uv_I = np.load('experiments/fig/etkf_cot_uv_I.npy')[:5,:,::2,::2]

mean_out_NN = np.load('experiments/fig/etkf_cot_mean_out_NN.npy')[:5,::2,::2]
mean_out_LK = np.load('experiments/fig/etkf_cot_mean_out_LK.npy')[:5,::2,::2]
mean_out_I = np.load('experiments/fig/etkf_cot_mean_out_I.npy')[:5,::2,::2]

#obs = obs[:,::3,::3]
#xmean = xmean[:,::3,::3]
#uv = uv[:,:,::3,::3]
fig = plot_time_steps(obs, [mean_out_NN, mean_out_LK, mean_out_I], [uv_NN, uv_LK, uv_I], ax, fig)

save()
