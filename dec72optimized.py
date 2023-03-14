# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:35:18 2023

@author: janus
"""
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import Normalize

# load data:
ds = xr.open_dataset('C:/Users/janus/Desktop/DTU/4. Semester/30110 Fagprojekt/NEMS_colocated/Nimbus5-NEMS_L2_1972m1217t011144_DR24_era5.nc')

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
        else:
            colors.append('blue')
    colors_list.append(colors)
    
for i in range(14):
    globals()[f'sizes{i}'] = [0.0001 if c == 'blue' else 1 for c in colors_list[i]]

plt.figure(1)
plt.title('DEC72 TB uncorrected 22.235 GHz vs 31.4 GHz with siconc values')
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')

for i in range(14):
    plt.scatter(globals()[f'a{i}chn2'], globals()[f'a{i}chn1'], s=globals()[f'sizes{i}'], c=colors_list[i])
plt.xlim([100, 330])
plt.ylim([100, 330])
plt.show(1)

for i in range(16):
    globals()[f'LAT{i}'] = ds.LAT[:, i]
    globals()[f'LON{i}'] = ds.LON[:, i]
    globals()[f'sizesLATLON{i}'] = [0.01 if c == 'blue' else 0.1 for c in colors_list[i]]
    
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
for i in range(16):
    plt.scatter(globals()[f'LON{i}'], globals()[f'LAT{i}'], s=globals()[f'sizesLATLON{i}'], c=colors_list[i])

plt.show(2)