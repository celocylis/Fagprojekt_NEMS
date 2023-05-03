import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import scipy
from scipy.interpolate import griddata
import pickle

# load data:
#ds = xr.open_dataset('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/NEMS_colocated/Nimbus5-NEMS_L2_1973m0622t063825_DR26_era5.nc')

# for pickle filer
with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/Siconc1nhvermaanedalledata.pkl','rb') as f:
    ds = pickle.load(f)


data = sum(ds,[])

data = np.array(data)


# extract data
lons = data[:,3]
lats = data[:,2]
siconc = data[:,0]

#%%

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(-90, 90, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), siconc, (xi, yi), method='linear')

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
plt.clim(0,1)

plt.show()

#%%
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.scatter(lons,lats,siconc)