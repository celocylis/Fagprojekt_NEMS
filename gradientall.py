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
list_colors = [(1,0,0),
               (1,0,0),
               (1,0,0),
               (1,0,0),
               (1,0,0),
               (1,0,0),
               (1,0,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,0,1),
               (0,0,1),
               (0,0,1),
               (0,0,1),
               (0,0,1),
               (0,0,1),
               (0,0,1)]

my_cmap = LinearSegmentedColormap.from_list('my_list', list_colors,N=21)


#%%

# for pickle filer
with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/GRDec.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
GR = data[:,2]

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(40, 80, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), GR, (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

#North pole:
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor=[0.3,0.3,0.3]))
ax.set_extent([0, 360, 90, 60], ccrs.PlateCarree())

ax.coastlines()

plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()
cbar.set_label('Gradient',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(-0.03,0.03)
plt.title("December 1972, gradient Nordpol",fontsize=20)

plt.show()

#%%

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(-80, -40, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), GR, (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

# south pole:
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor=[0.3,0.3,0.3]))
ax.set_extent([0, 360, -90, -60], ccrs.PlateCarree())

ax.coastlines()

plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()
cbar.set_label('Gradient',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(-0.03,0.03)
plt.title("December 1972, gradient Sydpol",fontsize=20)


plt.show()

#%%

# for pickle filer
with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/GRJun.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
GR = data[:,2]

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(40, 80, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), GR, (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

#North pole:
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor=[0.3,0.3,0.3]))
ax.set_extent([0, 360, 90, 60], ccrs.PlateCarree())

ax.coastlines()

plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()
cbar.set_label('Gradient',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(-0.03,0.03)
plt.title("Juni 1973, gradient Nordpol",fontsize=20)

plt.show()

#%%

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(-80, -40, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), GR, (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

# south pole:
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor=[0.3,0.3,0.3]))
ax.set_extent([0, 360, -90, -60], ccrs.PlateCarree())

ax.coastlines()

plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()
cbar.set_label('Gradient',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(-0.03,0.03)
plt.title("Juni 1973, gradient Sydpol",fontsize=20)


plt.show()
