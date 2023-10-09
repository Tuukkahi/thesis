import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pgfutils import setup_figure, save
setup_figure(width=0.5, height=0.3)

qopts = { 'angles': 'xy', 'scale_units': 'xy', 'scale': 0.5, 'color': 'lightgrey', 'minlength': 0}
def warp(X, u, v, dx=1.0, dy=1.0, dt=1.0):
    ny, nx = X.shape
    xi, yi = np.meshgrid(np.arange(nx), np.arange(ny))
    i = (xi - u*dt/dx).astype(int)
    j = (yi - v*dt/dy).astype(int)
    i = np.maximum(np.minimum(i, nx - 1), 0)
    j = np.maximum(np.minimum(j, ny - 1), 0)
    y = X.reshape(ny, nx)[j, i]
    return y

def formatter(val, pos):
    return str(int(val * 0.5))

ny = 10
nx = 10

x = np.arange(1,nx)
y = np.arange(1,ny)
mx, my = np.meshgrid(x,y)

d1 = np.zeros((ny,nx))
u = np.ones((ny,nx))
v = np.ones((ny,nx))
d1[2:4,2:4] = 1
d2 = 1.5*warp(d1,2*u,2*v)
d3 = 1.5*warp(d2,2*u,2*v)
plt.pcolormesh(d1+d2+d3, cmap='binary')

plt.gca().xaxis.set_major_formatter(FuncFormatter(formatter))
plt.gca().yaxis.set_major_formatter(FuncFormatter(formatter))

plt.grid()
u[-2,:] = 0
v[-2,:] = 0
u[:,-2] = 0
v[:,-2] = 0
plt.quiver(mx[::2,::2],my[::2,::2], u[::2,::2],v[::2,::2], **qopts)

save()
