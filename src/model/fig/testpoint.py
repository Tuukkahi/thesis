import matplotlib.pyplot as plt
import numpy as np
from pgfutils import setup_figure, save
setup_figure(width=0.5, height=0.3)

points = [[np.cos(phi), np.sin(phi)] for phi in np.linspace(2*np.pi, 0, 6)[:-1]]
points = np.vstack([np.zeros((1,2)), points])
fig, ax = plt.subplots()
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.scatter(points[:,0],points[:,1], color='black')

for i in range(6):
    text = "$(x_{" + str(i) + "}, y_{" + str(i) + "})$"
    ax.annotate(text, (points[i,0]+0.03, points[i,1]-0.11))

ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect='equal')
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.grid(which='both',axis='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)

save()
