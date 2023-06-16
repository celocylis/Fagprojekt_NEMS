# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:22:45 2023

@author: janus
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import glob
import pickle
import math

filelist = glob.glob('C:/Users/janus/Desktop/DTU/4. Semester/30110 Fagprojekt/NEMS_colocated/*.nc')
dataset = []
for file in filelist:
    dataset.append(xr.open_dataset(file))

dec72=[dataset[0],dataset[1],dataset[2]]
jun73=[dataset[20],dataset[21],dataset[22]]

dec72vaerdier = []

for x in range (len(dec72)):
    for i in range(len(dec72[x].siconc)):
        for j in range(14):
            chn31 = float(dec72[x].TBNEMS[i,j,1])
            chn22 = float(dec72[x].TBNEMS[i,j,0])
            sic = float(dec72[x].siconc[i,j])
            if math.isnan(sic)==False :
                dec72vaerdier.append([chn31,chn22,sic])
            print(i)
dec72vaerdier = np.array(dec72vaerdier)

jun73vaerdier = []

for x in range (len(jun73)):
    for i in range(len(jun73[x].siconc)):
        for j in range(14):
            chn31 = float(jun73[x].TBNEMS[i,j,1])
            chn22 = float(jun73[x].TBNEMS[i,j,0])
            sic = float(jun73[x].siconc[i,j])
            if math.isnan(sic)==False :
                jun73vaerdier.append([chn31,chn22,sic])
            print(i)
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

sizesdec72 = [0.001 if c == 'blue' else 0.5 for c in colorsdec72]
sizesjun73 = [0.001 if c == 'blue' else 0.5 for c in colorsjun73]


plt.scatter(dec72vaerdier[:,1],dec72vaerdier[:,0],s=sizesdec72,c=colorsdec72)
plt.title("December 72")
plt.xlim([100, 330])
plt.ylim([100, 330])
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')
plt.show(1)

plt.scatter(jun73vaerdier[:,1],jun73vaerdier[:,0],s=sizesjun73,c=colorsjun73)
plt.title("Juni 73")
plt.xlim([100, 330])
plt.ylim([100, 330])
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')
plt.show(2)