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
with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/RTM1chnDec22N.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
siconc = data[:,2]

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
plt.title("December 1972, 22GHz RTM, Nordpol",fontsize=17.5)

plt.show()

#%%

with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/RTM1chnDec22S.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
siconc = data[:,2]


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
plt.title("December 1972, 22GHz RTM, Sydpol",fontsize=17.5)


plt.show()

#%%
with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/RTM1chnDec31N.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
siconc = data[:,2]

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
plt.title("December 1972, 31GHz RTM, Nordpol",fontsize=17.5)

plt.show()

#%%

with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/RTM1chnDec31S.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
siconc = data[:,2]


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
plt.title("December 1972, 31GHz RTM, Sydpol",fontsize=17.5)


plt.show()

#%%
with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/RTM1chnJun22N.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
siconc = data[:,2]

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
plt.title("Juni 1973, 22GHz RTM, Nordpol",fontsize=20)

plt.show()

#%%

with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/RTM1chnJun22S.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
siconc = data[:,2]


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
plt.title("Juni 1973, 22GHz RTM, Sydpol",fontsize=20)


plt.show()

#%%
with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/RTM1chnJun31N.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
siconc = data[:,2]

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
plt.title("Juni 1973, 31GHz RTM, Nordpol",fontsize=20)

plt.show()

#%%

with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/RTM1chnJun31S.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
siconc = data[:,2]


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
plt.title("Juni 1973, 31GHz RTM, Sydpol",fontsize=20)


plt.show()

#%%

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(-90, 90, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), siconc, (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 8))

# Whole earth:
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))

# south pole:
#ax = plt.axes(projection=ccrs.SouthPolarStereo())
#ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
#ax.set_extent([0, 360, -90, -60], ccrs.PlateCarree())

#North pole:
#ax = plt.axes(projection=ccrs.NorthPolarStereo())
#ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
#ax.set_extent([0, 360, 90, 60], ccrs.PlateCarree())

ax.coastlines()


plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
plt.colorbar()
plt.clim(0,1)


plt.show()

#%%
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.coastlines()
plt.scatter(lons,lats,siconc)
