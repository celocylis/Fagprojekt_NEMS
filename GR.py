"""
Created on Mon Mar 13 11:13:53 2023

@author: noahe
"""

import matplotlib.pyplot as plt
import xarray as xr
import cartopy 
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#Indlæs data
ds = xr.open_dataset('C:/Users/noahe/OneDrive/Dokumenter/DTU/Fagprojekt/NEMS_colocated/Nimbus5-NEMS_L2_1972m1217t011144_DR24_era5.nc')

#Udregn gradient mellem de to TB temperaturer
GR = (ds.TBNEMS[:,0,1]-ds.TBNEMS[:,0,0])/(ds.TBNEMS[:,0,0]+ds.TBNEMS[:,0,1]) 

#Extract longitude and latitude data from the xarray dataset
LON = ds.LON[:,1]
LAT = ds.LAT [:,1]


plt.figure(figsize=(24,8));
plt.plot(GR,'+')
plt.title('Gradient mellem Tb1 og Tb2', fontsize=20) 
plt.xlabel('Maling') 
plt.ylabel('Gradient') 


plt.figure(figsize=(24,16));
ax = plt.axes(projection=cartopy.crs.NorthPolarStereo());

ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=12)


colormesh = ax.pcolormesh(LON, LAT, GR, transform=ccrs.PlateCarree());
cbar = plt.colorbar(colormesh)
cbar.set_label('GR',fontsize=20)
cbar.ax.tick_params(labelsize=16)


ax.coastlines();

from shapely.geometry.polygon import LinearRing
lons = [-20, -20, -48, -48]
lats = [42, 55, 55, 42]
ring = LinearRing(list(zip(lons, lats)))

plt.title('GR uncorrected \n DATO', fontsize=20)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.top_labels  = True
gl.left_labels  = True
gl.xlines = True
gl.xlocator = mticker.Fixed
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER