# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:58:49 2023

@author: janus
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import glob
import pickle

#%%
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
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

with open('nordsiconcchn1decjun.pkl', 'rb') as f:
    sicn1decjun = pickle.load(f)
with open('nordsiconcchn2decjun.pkl', 'rb') as f:
    sicn2decjun = pickle.load(f)
with open('sydsiconcchn1decjun.pkl', 'rb') as f:
    sics1decjun = pickle.load(f)
with open('sydsiconcchn2decjun.pkl', 'rb') as f:
    sics2decjun = pickle.load(f)

with open('nordsiconcchn1hverset.pkl', 'rb') as f:
    sicn1hverset = pickle.load(f)
with open('nordsiconcchn2hverset.pkl', 'rb') as f:
    sicn2hverset = pickle.load(f)
with open('sydsiconcchn1hverset.pkl', 'rb') as f:
    sics1hverset = pickle.load(f)
with open('sydsiconcchn2hverset.pkl', 'rb') as f:
    sics2hverset = pickle.load(f)
    
datadec = [sicn1decjun[0],sics1decjun[0],sicn2decjun[0],sics2decjun[0]]
datajun = [sicn1decjun[1],sics1decjun[1],sicn2decjun[1],sics2decjun[1]]
dec72chn1=np.array(datadec[0]+datadec[1])
dec72chn2=np.array(datadec[2]+datadec[3])
jun73chn1=np.array(datajun[0]+datajun[1])
jun73chn2=np.array(datajun[2]+datajun[3])

coldec72c1 = []
for x in range(len(dec72chn1)):
   if dec72chn1[x,2] >= 0.9:
       coldec72c1.append('red')
   elif dec72chn1[x,2] > 0.05:
       coldec72c1.append('green')
   else:
       coldec72c1.append('blue')

coldec72c2 = []
for x in range(len(dec72chn2)):
   if dec72chn2[x,2] >= 0.9:
       coldec72c2.append('red')
   elif dec72chn2[x,2] > 0.05:
       coldec72c2.append('green')
   else:
       coldec72c2.append('blue')

coljun73c1 = []
for x in range(len(jun73chn1)):
   if jun73chn1[x,2] >= 0.9:
       coljun73c1.append('red')
   elif jun73chn1[x,2] > 0.05:
       coljun73c1.append('green')
   else:
       coljun73c1.append('blue')

coljun73c2 = []
for x in range(len(jun73chn2)):
   if jun73chn2[x,2] >= 0.9:
       coljun73c2.append('red')
   elif jun73chn2[x,2] > 0.05:
       coljun73c2.append('green')
   else:
       coljun73c2.append('blue')


ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
sc=plt.scatter(dec72chn1[:,1],dec72chn1[:,0],s=0.05,c=dec72chn1[:,2],cmap=my_cmap)
plt.title("December 72 22Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(1)

ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([0,360,-90,-60],ccrs.PlateCarree())
ax.coastlines()
sc = plt.scatter(dec72chn1[:,1],dec72chn1[:,0],s=0.05,c=dec72chn1[:,2],transform=ccrs.PlateCarree(),cmap=my_cmap)
plt.title("December 72 Sydpol 22Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(2)

ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([0,360,60,90],ccrs.PlateCarree())
ax.coastlines()
sc=plt.scatter(dec72chn1[:,1],dec72chn1[:,0],s=0.05,c=dec72chn1[:,2],transform=ccrs.PlateCarree(),cmap=my_cmap)
plt.title("December 72 Nordpol 22Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(3)



ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
sc=plt.scatter(dec72chn2[:,1],dec72chn2[:,0],s=0.05,c=dec72chn2[:,2],cmap=my_cmap)
plt.title("December 72 31Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(4)

ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([0,360,-90,-60],ccrs.PlateCarree())
ax.coastlines()
sc=plt.scatter(dec72chn2[:,1],dec72chn2[:,0],s=0.05,c=dec72chn2[:,2],transform=ccrs.PlateCarree(),cmap=my_cmap)
plt.title("December 72 Sydpol 31Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(5)

ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([0,360,60,90],ccrs.PlateCarree())
ax.coastlines()
sc=plt.scatter(dec72chn2[:,1],dec72chn2[:,0],s=0.05,c=dec72chn2[:,2],transform=ccrs.PlateCarree(),cmap=my_cmap)
plt.title("December 72 Nordpol 31Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(6)


ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
sc=plt.scatter(jun73chn1[:,1],jun73chn1[:,0],s=0.05,c=jun73chn1[:,2],cmap=my_cmap)
plt.title("Juni 73 22Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(7)

ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([0,360,-90,-60],ccrs.PlateCarree())
ax.coastlines()
sc=plt.scatter(jun73chn1[:,1],jun73chn1[:,0],s=0.05,c=jun73chn1[:,2],transform=ccrs.PlateCarree(),cmap=my_cmap)
plt.title("Juni 73 Sydpol 22Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(8)

ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([0,360,60,90],ccrs.PlateCarree())
ax.coastlines()
sc=plt.scatter(jun73chn1[:,1],jun73chn1[:,0],s=0.05,c=jun73chn1[:,2],transform=ccrs.PlateCarree(),cmap=my_cmap)
plt.title("Juni 73 Nordpol 22Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(9)



ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
sc=plt.scatter(jun73chn2[:,1],jun73chn2[:,0],s=0.05,c=jun73chn2[:,2],cmap=my_cmap)
plt.title("Juni 73 31Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(10)

ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.set_extent([0,360,-90,-60],ccrs.PlateCarree())
ax.coastlines()
sc=plt.scatter(jun73chn2[:,1],jun73chn2[:,0],s=0.05,c=jun73chn2[:,2],transform=ccrs.PlateCarree(),cmap=my_cmap)
plt.title("Juni 73 Sydpol 31Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(11)

ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([0,360,60,90],ccrs.PlateCarree())
ax.coastlines()
sc=plt.scatter(jun73chn2[:,1],jun73chn2[:,0],s=0.05,c=jun73chn2[:,2],transform=ccrs.PlateCarree(),cmap=my_cmap)
plt.title("Juni 73 Nordpol 31Ghz")
cbar=plt.colorbar(sc)
cbar.set_label('Iskoncentration')
plt.clim(0,1)
plt.show(12)



dec72chn1=np.array(datadec[0]+datadec[1])
dec72chn2=np.array(datadec[2]+datadec[3])
jun73chn1=np.array(datajun[0]+datajun[1])
jun73chn2=np.array(datajun[2]+datajun[3])

with open('dec72chn1TILGRID.pkl', 'wb') as f:
    pickle.dump(dec72chn1, f)
with open('dec72chn2TILGRID.pkl', 'wb') as f:
    pickle.dump(dec72chn2, f)
with open('jun73chn1TILGRID.pkl', 'wb') as f:
    pickle.dump(jun73chn1, f)
with open('jun73chn2TILGRID.pkl', 'wb') as f:
    pickle.dump(jun73chn2, f)