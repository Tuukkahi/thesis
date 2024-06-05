import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.ndimage import gaussian_filter as gf
from pgfutils import setup_figure, save
setup_figure(width=1, height=0.2)

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
mx, my = np.meshgrid(x,y)

V = vandermonde()
au = np.array([6, -0.5, -0.05, -0.5, -0.5, 0.1])
av = np.array([-8, -1, -1, 0.2, 0.5, 0.1])
u = (V @ au).reshape(128,128)
v = (V @ av).reshape(128,128)

image = 0.5 * np.exp(-0.004 * ((mx.ravel() - 30)**2 + (my.ravel() - 100)**2))
image[image<0.04] = 0.0
image += 0.5 * np.exp(-0.002 * ((mx.ravel() - 80)**2 + (my.ravel() - 60)**2))
image[image<0.04] = 0.0
image = image.reshape((ny,nx))

mx = mx/127
my = my/127


fig, ax = plt.subplots(1,3)
ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)
ax[2].set_box_aspect(1)

ax[0].contourf(mx,my, image, cmap='binary', extend='both')
ax[0].quiver(mx[::15,::15],my[::15,::15], u[::15,::15],v[::15,::15], alpha=0.7, width=0.005)
ax[0].set_xlabel("$t=0$")
ax[0].set_yticks([])
ax[0].set_xticks([])

image = warp(image,u,v,D=5)
ax[1].contourf(mx,my, image, cmap='binary', extend='both')
ax[1].quiver(mx[::15,::15],my[::15,::15], u[::15,::15],v[::15,::15], alpha=0.7, width=0.005)
ax[1].set_xlabel("$t=1$")
ax[1].set_yticks([])
ax[1].set_xticks([])


image = warp(image,u,v,D=5)
ax[2].contourf(mx,my, image, cmap='binary', extend='both')
ax[2].quiver(mx[::15,::15],my[::15,::15], u[::15,::15],v[::15,::15], alpha=0.7, width=0.005)
ax[2].set_xlabel("$t=2$")
ax[2].set_xticks([])
ax[2].set_yticks([])

save()
