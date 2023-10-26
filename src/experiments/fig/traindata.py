import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from pgfutils import setup_figure, save
setup_figure(width=0.5, height=0.5)

test_data = np.load("experiments/fig/test_data_values.npy")
lat = np.load("experiments/fig/test_data_lat_values.npy")
lon = np.load("experiments/fig/test_data_lon_values.npy")
test_data = test_data[::4,::4]
lat = lat[::4,::4]
lon = lon[::4,::4]

def create_polygon(coords):
  pol = [[np.min(coords[0]), np.min(coords[1])],
  [np.max(coords[0]), np.min(coords[1])],
  [np.max(coords[0]), np.max(coords[1])],
  [np.min(coords[0]), np.max(coords[1])]]
  #[np.min(coords[0]), np.min(coords[1])]]
  return pol

latex_width_pt = 427
width_in = latex_width_pt / 72.27
height_in = width_in
fig, ax = plt.subplots(figsize=(width_in, 0.6*height_in), subplot_kw={'projection': ccrs.PlateCarree()})
test_data[np.isnan(test_data)] = 1e10
im = plt.pcolormesh(lon, lat, test_data, transform=ccrs.PlateCarree(), cmap="bone_r", vmin=0,vmax=20)
cb = plt.colorbar(im, ax = ax)
cb.ax.set_ylabel("COT value")
#cb.ax.tick_params(labelsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())
#ax.tick_params(labelsize=15)
ax.coastlines(resolution='10m')
ax.coastlines(color='black')

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ext = [lon[-1].min(), lon[-1].max(), 
               lat[:,-1].min(), lat[:,0].max()]
ax.set_extent(ext, crs=ccrs.PlateCarree())
ax.set_aspect((lon.max() - lon.min()) / (lat.max() - lat.min()) * 0.8)

latlon_blocks = [
    [[ext[3], np.mean([ext[2], ext[3]])], [ext[0], np.mean([ext[0], ext[1]])]],
    [[ext[2], np.mean([ext[2], ext[3]])], [ext[0], np.mean([ext[0], ext[1]])]],
    [[ext[3], np.mean([ext[2], ext[3]])], [ext[1], np.mean([ext[0], ext[1]])]],
    [[ext[2], np.mean([ext[2], ext[3]])], [ext[1], np.mean([ext[0], ext[1]])]],
]

for i,coords in enumerate(latlon_blocks):
  pol = create_polygon(coords)
  ys, xs = zip(*pol)
  plt.plot(xs,ys,color="black")
  plt.text(np.mean(xs),np.mean(ys),i+1,horizontalalignment='center',verticalalignment='center')
plt.tight_layout()
fig.set_size_inches(width_in, 0.6*height_in)
save()
