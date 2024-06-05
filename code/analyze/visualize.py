import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def plot_time_steps(actual, predict, flow_vectors=None, vlim=[-5,5]):

  lags = actual.shape[0] - predict.shape[0]
  actual = actual.cpu().detach().numpy()
  predict = predict.cpu().detach().numpy()

  #vmin = np.min([np.min(actual), np.min(predict)])
  #vmax = np.min([np.max(actual), np.max(predict)])
  vmin = vlim[0]
  vmax = vlim[1]

  if flow_vectors is not None:
    assert predict.shape[0] == flow_vectors.shape[0]
    flow_vectors = flow_vectors.cpu().detach().numpy()
    fig, ax = plt.subplots(2,
                           actual.shape[0],
                          figsize=(14, 7))
    x_len = flow_vectors.shape[-2]
    y_len = flow_vectors.shape[-1]
    grid_x, grid_y = np.meshgrid(np.linspace(0, x_len - 1, x_len),
                                 np.linspace(0, y_len - 1, y_len))
  else:
    fig, ax = plt.subplots(2,
                           actual.shape[0],
                           figsize=(10 * actual.shape[0], 20))

  for i in range(actual.shape[0]):
    ax[0, i].imshow(actual[i], vmin=vmin, vmax=vmax, origin="lower")
    if i < lags - 1:
      ax[0, i].set_title("$Y_{t-" + str(lags - i + 1) + "}$")
    elif i == lags - 1:
      ax[0, i].set_title("$Y_{t}$")
    else:
      ax[0, i].set_title("$Y_{t+" + str(i - lags + 1) + "}$")

  for i in range(predict.shape[0] + lags):
    if i < lags:
      ax[1, i].axis("off")
    else:
      ax[1, i].set_title("$\widehat{Y}_{t+" + str(i - lags + 1) + "}$")
      im = ax[1, i].imshow(predict[i - lags],
                           vmin=vmin,
                           vmax=vmax,
                           origin="lower")
      if flow_vectors is not None:
        ax[1, i].quiver(grid_x[::8,::8], grid_y[::8,::8], flow_vectors[i - lags, 0][::8,::8],
                        flow_vectors[i - lags, 1][::8,::8], alpha=0.3)

  fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.5)
  return fig


def plot_flow(image: Tensor, flow_vectors: Tensor):
  image = image.cpu().detach().numpy()
  flow_vectors = flow_vectors.cpu().detach().numpy()

  x_len = image.shape[0]
  y_len = image.shape[1]
  grid_x, grid_y = np.meshgrid(np.linspace(0, x_len - 1, x_len),
                               np.linspace(0, y_len - 1, y_len))

  fig, ax = plt.subplots(figsize=(15, 15))

  ax.imshow(image, origin="lower")
  ax.quiver(grid_x, grid_y, flow_vectors[0], flow_vectors[1])

  return fig
