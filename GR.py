# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:55:32 2023

@author: noahe
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# load data:
ds = xr.open_dataset('C:/Users/noahe/OneDrive/Dokumenter/DTU/Fagprojekt/NEMS_colocated/Nimbus5-NEMS_L2_1972m1217t011144_DR24_era5.nc')
#ds = xr.open_dataset('C:/Users/noahe/OneDrive/Dokumenter/DTU/Fagprojekt/NEMS_colocated/Nimbus5-NEMS_L2_1973m0617t031041_DR26_era5.nc')



colors_list = []

for i in range(14):
    colors = []
    GR = (ds.TBNEMS[:,i,1]-ds.TBNEMS[:,i,0])/(ds.TBNEMS[:,i,0]+ds.TBNEMS[:,i,1]) 
    
    for value in GR:
        if value > 0.01:
            colors.append('blue')
        elif value <= -0.1:
            colors.append('black')
        else:
            colors.append('green')
    colors_list.append(colors)
    
plt.figure(1)
for i in range(16):
    globals()[f'LAT{i}'] = ds.LAT[:, i]
    globals()[f'LON{i}'] = ds.LON[:, i]

#North pole
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 90, 60], ccrs.PlateCarree())

ax.coastlines()

for i in range(14):
    plt.scatter(globals()[f'LON{i}'], globals()[f'LAT{i}'], s=0.1, c=colors_list[i],transform=ccrs.PlateCarree())
plt.show(1)

plt.figure(2)
for i in range(16):
    globals()[f'LAT{i}'] = ds.LAT[:, i]
    globals()[f'LON{i}'] = ds.LON[:, i]

#South pole
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())

ax.coastlines()

for i in range(14):
    plt.scatter(globals()[f'LON{i}'], globals()[f'LAT{i}'], s=0.1, c=colors_list[i],transform=ccrs.PlateCarree())


plt.show(2)