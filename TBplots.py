import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import numpy as np
import scipy
from scipy.interpolate import griddata
import pickle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


#%%
# create colormap
list_colors = [(0,0,.6),
               (0,0,.6),
               (0,0,.6),
               (0,0,.6),
               (0,0,1),
               (0,0,1),
               (0,0,1),
               (0,0,1),
               (0,.4,0),
               (0,.4,0),
               (0,.4,0),
               (0,.4,0),
               (0,.4,0),
               (0,.4,0),
               (0,.4,0),
               (0,.8,0),
               (0,.8,0),
               (1,0,0),
               (1,0,0),
               (1,0,0)]

my_cmap = LinearSegmentedColormap.from_list('my_list', list_colors,N=20)


#%%

# for pickle filer
with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/dec72chn1TILGRID.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))

# extract data
lons = data[:,1]
lats = data[:,0]
TB = data[:,3]

#%%

fig = plt.figure(figsize=(8, 6))

#North pole:
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, 90, 60], ccrs.PlateCarree())

ax.coastlines()

plt.scatter(lons, lats,c=TB, s=2, cmap='rainbow', transform=ccrs.PlateCarree())
cbar=plt.colorbar()
cbar.set_label('TB',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(130,250)
plt.title("December 1972, TB, Nordpol",fontsize=20)

plt.show()

#%%

fig = plt.figure(figsize=(8, 6))

#North pole:
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, -60, -90], ccrs.PlateCarree())

ax.coastlines()

plt.scatter(lons, lats,c=TB, s=2, cmap='rainbow', transform=ccrs.PlateCarree())
cbar=plt.colorbar()
cbar.set_label('TB',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(130,250)
plt.title("December 1972, TB, Sydpol",fontsize=20)

plt.show()