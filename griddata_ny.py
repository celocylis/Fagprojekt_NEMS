import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import Normalize
import scipy
from scipy.interpolate import griddata


# load data:
ds = xr.open_dataset('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/NEMS_colocated/Nimbus5-NEMS_L2_1973m0622t063825_DR26_era5.nc')

for i in range(14):
    globals()[f'a{i}chn1'] = ds.TBNEMS[:, i, 0]
    globals()[f'a{i}chn2'] = ds.TBNEMS[:, i, 1] 

for i in range(16):
    globals()[f'siconc{i}'] = ds.siconc[:, i]
    globals()[f'siconc_norm{i}'] = Normalize(vmin=siconc0.min(), vmax=siconc0.max())(siconc0)

siconc_norm_list = [siconc_norm0, siconc_norm1, siconc_norm2, siconc_norm3, siconc_norm4, siconc_norm5, siconc_norm6, siconc_norm7, siconc_norm8, siconc_norm9, siconc_norm10, siconc_norm11, siconc_norm12, siconc_norm13, siconc_norm14, siconc_norm15]
colors_list = []


for siconc_norm in siconc_norm_list:
    colors = []
    for value in siconc_norm:
        if value >= 0.9:
            colors.append('red')
        elif value > 0:
            colors.append('green')
        elif value <= 0:
            colors.append('blue')
        else:
            colors.append('black')
    colors_list.append(colors)



plt.figure(2)
for i in range(16):
    globals()[f'LAT{i}'] = ds.LAT[:, i]
    globals()[f'LON{i}'] = ds.LON[:, i]
    globals()[f'sizesLATLON{i}'] = [0.01 if c == 'blue' else 0.1 for c in colors_list[i]]

#Mercator projection
ax = plt.axes(projection=ccrs.PlateCarree())

ax.coastlines()

for i in range(16):
    plt.scatter(globals()[f'LON{i}'], globals()[f'LAT{i}'], s=globals()[f'sizesLATLON{i}'], c=colors_list[i],transform=ccrs.PlateCarree())
plt.show(2)


'''
plt.figure(3)
for i in range(16):
    globals()[f'LAT{i}'] = ds.LAT[:, i]
    globals()[f'LON{i}'] = ds.LON[:, i]
    globals()[f'sizesLATLON{i}'] = [0.01 if c == 'blue' else 0.1 for c in colors_list[i]]

#North pole
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 90, 60], ccrs.PlateCarree())

ax.coastlines()

for i in range(16):
    plt.scatter(globals()[f'LON{i}'], globals()[f'LAT{i}'], s=globals()[f'sizesLATLON{i}'], c=colors_list[i],transform=ccrs.PlateCarree())

plt.show(3)
'''

plt.figure(4)
for i in range(16):
    globals()[f'LAT{i}'] = ds.LAT[:, i]
    globals()[f'LON{i}'] = ds.LON[:, i]
    globals()[f'sizesLATLON{i}'] = [0.01 if c == 'blue' else 0.1 for c in colors_list[i]]

#South pole
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

ax.coastlines()

for i in range(16):
    plt.scatter(globals()[f'LON{i}'], globals()[f'LAT{i}'], s=globals()[f'sizesLATLON{i}'], c=colors_list[i],transform=ccrs.PlateCarree())


plt.show(4)



#%%

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import Normalize
import scipy
from scipy.interpolate import griddata

# load data:
ds = xr.open_dataset('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/NEMS_colocated/Nimbus5-NEMS_L2_1973m0622t063825_DR26_era5.nc')

# extract data
lons = ds.LON.values.flatten()
lats = ds.LAT.values.flatten()
siconc = ds.siconc.values.flatten()


# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(-90, 90, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), siconc, (xi, yi), method='nearest')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 8))

# Whole earth:
#ax = plt.axes(projection=ccrs.PlateCarree())

# south pole:
#ax = plt.axes(projection=ccrs.SouthPolarStereo())
#ax.set_extent([0, 360, -90, -60], ccrs.PlateCarree())

#North pole:
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([0, 360, 90, 60], ccrs.PlateCarree())

ax.coastlines()

plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap='jet')
plt.colorbar()

plt.show()


