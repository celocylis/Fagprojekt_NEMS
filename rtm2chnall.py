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
list_colors = [(0,0,1),
               (0,0,1),
               (0,0,1),
               (0,0,1),
               (0,0,1),
               (0,0,1),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (0,1,0),
               (1,0,0),
               (1,0,0)]

my_cmap = LinearSegmentedColormap.from_list('my_list', list_colors,N=20)


#%%
with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/RTMfcomisoDec.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
siconc = data[:,2]

#%%
# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(50, 80, 1000))

# interpolate the data onto the grid
zi = griddata((lons, lats), siconc, (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

#North pole:
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, 90, 60], ccrs.PlateCarree())

ax.coastlines()

img = plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()
cbar.set_label('Iskoncentration',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(0,1)
plt.title("December 1972, 2chn RTM, Nordpol",fontsize=18)

plt.show()

#%%
boi = img.get_array()

boi[boi > 1] = 1
boi[boi < 0] = 0

inshallah = np.mean(boi)

#%%

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(-80, -50, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), siconc, (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

# south pole:
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, -90, -60], ccrs.PlateCarree())

ax.coastlines()

plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()
cbar.set_label('Iskoncentration',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(0,1)
plt.title("December 1972, 2chn RTM, Sydpol",fontsize=18)


plt.show()

#%%
with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/RTMfcomisoJun.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
siconc = data[:,2]
#%%
# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(50, 80, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), siconc, (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

#North pole:
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, 90, 60], ccrs.PlateCarree())

ax.coastlines()

plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()
cbar.set_label('Iskoncentration',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(0,1)
plt.title("Juni 1973, 2chn RTM, Nordpol",fontsize=20)

plt.show()

#%%

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(-80, -50, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), siconc, (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

# south pole:
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, -90, -60], ccrs.PlateCarree())

ax.coastlines()

plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()
cbar.set_label('Iskoncentration',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(0,1)
plt.title("Juni 1973, 2chn RTM, Sydpol",fontsize=20)


plt.show()

#%%

data1 = np.array(filtered_data,dtype = np.dtype('float'))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([0, 360, -90, -60], ccrs.PlateCarree())
ax.coastlines()
plt.scatter(data1[:,1], data1[:,0], data1[:,2],transform=ccrs.PlateCarree())

#%%
data1 = np.array(filtered_data,dtype = np.dtype('float'))
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([0, 360, -90, -60], ccrs.PlateCarree())
ax.coastlines()
plt.scatter(data[:,1], data[:,0], data[:,2],transform=ccrs.PlateCarree())

