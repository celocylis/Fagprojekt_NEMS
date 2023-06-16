# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:46:51 2023

@author: janus
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import glob
import pickle

filelist = glob.glob('C:/Users/janus/Desktop/DTU/4. Semester/30110 Fagprojekt/NEMS_colocated/*.nc')
dataset = []
for file in filelist:
    dataset.append(xr.open_dataset(file))


dec72=[dataset[0],dataset[1],dataset[2]]
jun73=[dataset[20],dataset[21],dataset[22]]

dec72vaerdier = []

jun73vaerdier = []

for x in range (len(dec72)):
    for i in range(len(dec72[x].siconc)):
        for j in range(14):
            lat = float(dec72[x].LAT[i,j])
            lon = float(dec72[x].LON[i,j])
            sic = float(dec72[x].siconc[i,j])
            dec72vaerdier.append([lat,lon,sic])
dec72vaerdier = np.array(dec72vaerdier)

for x in range (len(jun73)):
    for i in range(len(jun73[x].siconc)):
        for j in range(14):
            lat = float(jun73[x].LAT[i,j])
            lon = float(jun73[x].LON[i,j])
            sic = float(jun73[x].siconc[i,j])
            jun73vaerdier.append([lat,lon,sic])
jun73vaerdier = np.array(jun73vaerdier)

colorsdec72 = []
for x in range(len(dec72vaerdier)):
   if dec72vaerdier[x][2] >= 0.9:
       colorsdec72.append('red')
   elif dec72vaerdier[x][2] > 0.05:
       colorsdec72.append('green')
   else:
       colorsdec72.append('blue')

colorsjun73 = []
for x in range(len(jun73vaerdier)):
   if jun73vaerdier[x][2] >= 0.9:
       colorsjun73.append('red')
   elif jun73vaerdier[x][2] > 0.05:
       colorsjun73.append('green')
   else:
       colorsjun73.append('blue')


ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.scatter(dec72vaerdier[:,1],dec72vaerdier[:,0],s=0.05,c=colorsdec72)
plt.title("December 72")
plt.show(1)

ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([0,360,-90,-60],ccrs.PlateCarree())
ax.coastlines()
plt.scatter(dec72vaerdier[:,1],dec72vaerdier[:,0],s=0.05,c=colorsdec72,transform=ccrs.PlateCarree())
plt.title("December 72 Sydpol")
plt.show(2)

ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([0,360,60,90],ccrs.PlateCarree())
ax.coastlines()
plt.scatter(dec72vaerdier[:,1],dec72vaerdier[:,0],s=0.05,c=colorsdec72,transform=ccrs.PlateCarree())
plt.title("December 72 Nordpol")
plt.show(3)
    
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.scatter(jun73vaerdier[:,1],jun73vaerdier[:,0],s=0.05,c=colorsjun73)
plt.title("Juni 73")
plt.show(4)

ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([0,360,-90,-60],ccrs.PlateCarree())
ax.coastlines()
plt.scatter(jun73vaerdier[:,1],jun73vaerdier[:,0],s=0.05,c=colorsjun73,transform=ccrs.PlateCarree())
plt.title("Juni 73 Sydpol")
plt.show(5)

ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([0,360,60,90],ccrs.PlateCarree())
ax.coastlines()
plt.scatter(jun73vaerdier[:,1],jun73vaerdier[:,0],s=0.05,c=colorsjun73,transform=ccrs.PlateCarree())
plt.title("Juni 73 Nordpol")
plt.show(6)


