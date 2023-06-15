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
# for pickle filer
with open('C:/Users/cfmcf/OneDrive/4. DTU/4. semester/Fagprojekt/GRDec.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds,dtype = np.dtype('float'))


# extract data
lons = data[:,1]
lats = data[:,0]
GR = data[:,2]
siconc = data[:,3]

#%%
istypedata = []
for x in range(len(siconc)):
    if siconc[x] <= 0.3:
        istypedata.append([lats[x],lons[x],0])
    elif siconc[x] > 0.3:
        if GR[x] >= -0.02:
            istypedata.append([lats[x],lons[x],1])
        elif GR[x] < -0.02:
            istypedata.append([lats[x],lons[x],2])
        
istypedata = np.array(istypedata)

#%%
# create colormap
list_colors = [(0,0,1),
               (0,1,0),
               (1,0,0)]

my_cmap = LinearSegmentedColormap.from_list('my_list', list_colors,N=3)

#%%

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(40, 80, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), istypedata[:,2], (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

#North pole:
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, 90, 60], ccrs.PlateCarree())

ax.coastlines()

colormesh = plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()

cbar = fig.colorbar(colormesh, ticks=[0.33, 1, 1.66])
cbar.ax.set_yticklabels(['OW', 'MYI', 'FYI'])

cbar.set_label('istype',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(0,2)
plt.title("Gradient December 1972 nordpol",fontsize=17.5)

plt.show()

#%%

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(-80, -40, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), istypedata[:,2], (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

#South pole:
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, -60, -90], ccrs.PlateCarree())

ax.coastlines()

colormesh = plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()

cbar = fig.colorbar(colormesh, ticks=[0.33, 1, 1.66])
cbar.ax.set_yticklabels(['OW', 'MYI', 'FYI'])

cbar.set_label('istype',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(0,2)
plt.title("Gradient December 1972 sydpol",fontsize=17.5)

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
siconc = data[:,3]

#%%
istypedata = []
for x in range(len(siconc)):
    if siconc[x] <= 0.3:
        istypedata.append([lats[x],lons[x],0])
    elif siconc[x] > 0.3:
        if GR[x] >= -0.02:
            istypedata.append([lats[x],lons[x],2])
        elif GR[x] < -0.02:
            istypedata.append([lats[x],lons[x],1])
        
istypedata = np.array(istypedata)

#%%
# create colormap
list_colors = [(0,0,1),
               (0,1,0),
               (1,0,0)]

my_cmap = LinearSegmentedColormap.from_list('my_list', list_colors,N=3)

#%%

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(40, 80, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), istypedata[:,2], (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

#North pole:
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, 90, 60], ccrs.PlateCarree())

ax.coastlines()

colormesh = plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()

cbar = fig.colorbar(colormesh, ticks=[0.33, 1, 1.66])
cbar.ax.set_yticklabels(['OW', 'MYI', 'FYI'])

cbar.set_label('istype',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(0,2)
plt.title("Gradient Juni 1973 nordpol",fontsize=17.5)

plt.show()

#%%

# define the grid onto which we want to interpolate the data
xi, yi = np.meshgrid(np.linspace(0, 360, 1000), np.linspace(-80, -40, 1000))


# interpolate the data onto the grid
zi = griddata((lons, lats), istypedata[:,2], (xi, yi), method='linear')

# plot the interpolated data using Cartopy's pcolormesh function
fig = plt.figure(figsize=(8, 6))

#South pole:
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, -60, -90], ccrs.PlateCarree())

ax.coastlines()

colormesh = plt.pcolormesh(xi, yi, zi, transform=ccrs.PlateCarree(), cmap=my_cmap)
cbar=plt.colorbar()

cbar = fig.colorbar(colormesh, ticks=[0.33, 1, 1.66])
cbar.ax.set_yticklabels(['OW', 'MYI', 'FYI'])

cbar.set_label('istype',fontsize=15)
cbar.ax.tick_params(labelsize=16)
plt.clim(0,2)
plt.title("Gradient Juni 1973 sydpol",fontsize=17.5)

plt.show()