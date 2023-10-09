import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.ndimage import gaussian_filter as gf
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.7)

fig, ax = plt.subplots(4,5)


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

  vmin = np.min([np.min(actual), np.min(predict)])
  vmax = np.min([np.max(actual), np.max(predict)])

  if flow_vectors is not None:
    assert predict.shape[0] == flow_vectors.shape[0]
    x_len = flow_vectors.shape[-2]
    y_len = flow_vectors.shape[-1]
    grid_x, grid_y = np.meshgrid(np.linspace(0, x_len - 1, x_len),
                                 np.linspace(0, y_len - 1, y_len))
    grid_x /= x_len
    grid_y /= y_len

  for i in range(actual.shape[0]):
    ax[0, i].set_box_aspect(1)
    ax[0, i].contourf(grid_x, grid_y, actual[i], cmap='binary', extend='both')
    ax[0, i].set_yticks([])
    ax[0, i].set_xticks([])
    add_grid_lines(ax[0, i])
    if i < lags - 1:
      ax[0, i].set_title("$x_{t-" + str(lags - i - 1) + "}$")
    elif i == lags - 1:
      ax[0, i].set_title("$x_{t}$")
    else:
      ax[0, i].set_title("$x_{t+" + str(i - lags + 1) + "}$")

  for i in range(predict.shape[0] + lags):
    if i < lags:
      ax[1, i].axis("off")
    else:
      ax[1, i].set_yticks([])
      ax[1, i].set_xticks([])
      ax[1, i].set_box_aspect(1)
      ax[1, i].set_title("$\widehat{x}_{t+" + str(i - lags + 1) + "}$")
      ax[1, i].contourf(grid_x, grid_y, predict[i - lags], cmap='binary', extend='both')
      add_grid_lines(ax[1, i])
      if flow_vectors is not None:
        ax[1, i].quiver(grid_x[::16,::16], grid_y[::16,::16], flow_vectors[i - lags, 0][::16,::16],
                        flow_vectors[i - lags, 1][::16,::16], alpha=0.3, linewidth=4)

  return fig

def vandermonde():
  def intera(x):
    return np.stack([x[0,::], x[1,::], x[0,::]*x[1,::], x[0,::]**2, x[1,::]**2])

  V = np.stack(np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny)))
  V = intera(V)
  V = V.reshape(*V.shape[:-2], -1)
  V = np.r_[np.ones((1,V.shape[1])), V]
  return V.T

def warp(X, u, v, dx=1.0, dy=1.0, dt=1.0, D=0.0):
    ny, nx = X.shape
    xi, yi = np.meshgrid(np.arange(nx), np.arange(ny))
    i = (xi - u*dt/dx).astype(int)
    j = (yi - v*dt/dy).astype(int)
    i = np.maximum(np.minimum(i, nx - 1), 0)
    j = np.maximum(np.minimum(j, ny - 1), 0)
    if D > 0.0:
        y = gf(X.reshape(ny, nx), sigma=D)[j, i]
    else:
        y = X.reshape(ny, nx)[j, i]
    return y

nx = 128
ny = 128
x = np.arange(0,nx)
y = np.arange(0,ny)

V = vandermonde()
au = np.array([4, -0.5, -0.05, -0.5, -0.5, 0.1])
av = np.array([-6, -1, -1, 0.2, 0.5, 0.1])
u = (V @ au).reshape(128,128)
v = (V @ av).reshape(128,128)
mx, my = np.meshgrid(x,y)
image = 6 * np.exp(-0.004 * ((mx.ravel() - 30)**2 + (my.ravel() - 100)**2))
image[image<0.04] = 0.0
image = 0*image
image += 5 * np.exp(-0.003 * ((mx.ravel() - 60)**2 + (my.ravel() - 60)**2))
image[image<0.5] = 0.0
image = image.reshape((ny,nx))


obs = np.empty((5,128,128))
obs[0] = image.copy()
for i in range(4):
    obs[i+1] = warp(obs[i],u,v,D=0)
obs = (obs - obs.mean())/obs.std()

pred = np.load('experiments/fig/pred_ball.npy')
uv = np.load('experiments/fig/uv_ball.npy')

fig = plot_time_steps(obs, pred, np.array([uv,uv]), ax[:2,:], fig)

V = vandermonde()
au = np.array([-7, -0.5, -0.05, -0.5, -0.5, 0.1])
av = np.array([6, -1, -1, 0.2, 0.5, 0.1])
u = (V @ au).reshape(128,128)
v = (V @ av).reshape(128,128)
image = np.zeros((128,128))
image[70:90,40:60] = 4.0
image[20:50,80:110] = 4.0
obs = np.empty((5,128,128))
obs[0] = image.copy()
for i in range(4):
    obs[i+1] = warp(obs[i],u,v,D=0)
obs = (obs - obs.mean())/obs.std()

pred = np.load('experiments/fig/pred_blocks.npy')
uv = np.load('experiments/fig/uv_blocks.npy')
fig = plot_time_steps(obs, pred, np.array([uv,uv]), ax[2:,:], fig)

save()
