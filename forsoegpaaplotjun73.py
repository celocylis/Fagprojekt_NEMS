# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:53:34 2023

@author: janus
"""
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import Normalize

# load data:
ds = xr.open_dataset('C:/Users/janus/Desktop/DTU/4. Semester/30110 Fagprojekt/NEMS_colocated/Nimbus5-NEMS_L2_1973m0617t031041_DR26_era5.nc')

######################
a0chn1 = ds.TBNEMS[:, 0, 0]
a0chn2 = ds.TBNEMS[:, 0, 1]

a1chn1 = ds.TBNEMS[:, 1, 0]
a1chn2 = ds.TBNEMS[:, 1, 1]

a2chn1 = ds.TBNEMS[:, 2, 0]
a2chn2 = ds.TBNEMS[:, 2, 1]

a3chn1 = ds.TBNEMS[:, 3, 0]
a3chn2 = ds.TBNEMS[:, 3, 1]

a4chn1 = ds.TBNEMS[:, 4, 0]
a4chn2 = ds.TBNEMS[:, 4, 1]

a5chn1 = ds.TBNEMS[:, 5, 0]
a5chn2 = ds.TBNEMS[:, 5, 1]

a6chn1 = ds.TBNEMS[:, 6, 0]
a6chn2 = ds.TBNEMS[:, 6, 1]

a7chn1 = ds.TBNEMS[:, 7, 0]
a7chn2 = ds.TBNEMS[:, 7, 1]

a8chn1 = ds.TBNEMS[:, 8, 0]
a8chn2 = ds.TBNEMS[:, 8, 1]

a9chn1 = ds.TBNEMS[:, 9, 0]
a9chn2 = ds.TBNEMS[:, 9, 1]

a10chn1 = ds.TBNEMS[:, 10, 0]
a10chn2 = ds.TBNEMS[:, 10, 1]

a11chn1 = ds.TBNEMS[:, 11, 0]
a11chn2 = ds.TBNEMS[:, 11, 1]

a12chn1 = ds.TBNEMS[:, 12, 0]
a12chn2 = ds.TBNEMS[:, 12, 1]

a13chn1 = ds.TBNEMS[:, 13, 0]
a13chn2 = ds.TBNEMS[:, 13, 1]


siconc0 = ds.siconc[:, 0]
siconc1 = ds.siconc[:, 1]
siconc2 = ds.siconc[:, 2]
siconc3 = ds.siconc[:, 3]
siconc4 = ds.siconc[:, 4]
siconc5 = ds.siconc[:, 5]
siconc6 = ds.siconc[:, 6]
siconc7 = ds.siconc[:, 7]
siconc8 = ds.siconc[:, 8]
siconc9 = ds.siconc[:, 9]
siconc10 = ds.siconc[:, 10]
siconc11 = ds.siconc[:, 11]
siconc12 = ds.siconc[:, 12]
siconc13 = ds.siconc[:, 13]
siconc14 = ds.siconc[:, 14]
siconc15 = ds.siconc[:, 15]

# normalize siconc values
siconc_norm0 = Normalize(vmin=siconc0.min(), vmax=siconc0.max())(siconc0)
siconc_norm1 = Normalize(vmin=siconc1.min(), vmax=siconc1.max())(siconc1)
siconc_norm2 = Normalize(vmin=siconc2.min(), vmax=siconc2.max())(siconc2)
siconc_norm3 = Normalize(vmin=siconc3.min(), vmax=siconc3.max())(siconc3)
siconc_norm4 = Normalize(vmin=siconc4.min(), vmax=siconc4.max())(siconc4)
siconc_norm5 = Normalize(vmin=siconc5.min(), vmax=siconc5.max())(siconc5)
siconc_norm6 = Normalize(vmin=siconc6.min(), vmax=siconc6.max())(siconc6)
siconc_norm7 = Normalize(vmin=siconc7.min(), vmax=siconc7.max())(siconc7)
siconc_norm8 = Normalize(vmin=siconc8.min(), vmax=siconc8.max())(siconc8)
siconc_norm9 = Normalize(vmin=siconc9.min(), vmax=siconc9.max())(siconc9)
siconc_norm10 = Normalize(vmin=siconc10.min(), vmax=siconc10.max())(siconc10)
siconc_norm11 = Normalize(vmin=siconc11.min(), vmax=siconc11.max())(siconc11)
siconc_norm12 = Normalize(vmin=siconc12.min(), vmax=siconc12.max())(siconc12)
siconc_norm13 = Normalize(vmin=siconc13.min(), vmax=siconc13.max())(siconc13)
siconc_norm14 = Normalize(vmin=siconc14.min(), vmax=siconc14.max())(siconc14)
siconc_norm15 = Normalize(vmin=siconc15.min(), vmax=siconc15.max())(siconc15)

# create a color map for siconc values


# map siconc values to colors
colors0 = []
for value in siconc_norm0:
    if value >= 0.9:
        colors0.append('red')
    elif value > 0:
        colors0.append('green')
    else:
        colors0.append('blue')

colors1 = []
for value in siconc_norm1:
    if value >= 0.9:
        colors1.append('red')
    elif value > 0:
        colors1.append('green')
    else:
        colors1.append('blue')

colors2 = []
for value in siconc_norm2:
    if value >= 0.9:
        colors2.append('red')
    elif value > 0:
        colors2.append('green')
    else:
        colors2.append('blue')

colors3 = []
for value in siconc_norm3:
    if value >= 0.9:
        colors3.append('red')
    elif value > 0:
        colors3.append('green')
    else:
        colors3.append('blue')

colors4 = []
for value in siconc_norm4:
    if value >= 0.9:
        colors4.append('red')
    elif value > 0:
        colors4.append('green')
    else:
        colors4.append('blue')

colors5 = []
for value in siconc_norm5:
    if value >= 0.9:
        colors5.append('red')
    elif value > 0:
        colors5.append('green')
    else:
        colors5.append('blue')

colors6 = []
for value in siconc_norm6:
    if value >= 0.9:
        colors6.append('red')
    elif value > 0:
        colors6.append('green')
    else:
        colors6.append('blue')

colors7 = []
for value in siconc_norm7:
    if value >= 0.9:
        colors7.append('red')
    elif value > 0:
        colors7.append('green')
    else:
        colors7.append('blue')

colors8 = []
for value in siconc_norm8:
    if value >= 0.9:
        colors8.append('red')
    elif value > 0:
        colors8.append('green')
    else:
        colors8.append('blue')

colors9 = []
for value in siconc_norm9:
    if value >= 0.9:
        colors9.append('red')
    elif value > 0:
        colors9.append('green')
    else:
        colors9.append('blue')

colors10 = []
for value in siconc_norm10:
    if value >= 0.9:
        colors10.append('red')
    elif value > 0:
        colors10.append('green')
    else:
        colors10.append('blue')

colors11 = []
for value in siconc_norm11:
    if value >= 0.9:
        colors11.append('red')
    elif value > 0:
        colors11.append('green')
    else:
        colors11.append('blue')

colors12 = []
for value in siconc_norm12:
    if value >= 0.9:
        colors12.append('red')
    elif value > 0:
        colors12.append('green')
    else:
        colors12.append('blue')
        
colors13 = []
for value in siconc_norm13:
    if value >= 0.9:
        colors13.append('red')
    elif value > 0:
        colors13.append('green')
    else:
        colors13.append('blue')

colors14 = []
for value in siconc_norm14:
    if value >= 0.9:
        colors14.append('red')
    elif value > 0:
        colors14.append('green')
    else:
        colors14.append('blue')

colors15 = []
for value in siconc_norm15:
    if value >= 0.9:
        colors15.append('red')
    elif value > 0:
        colors15.append('green')
    else:
        colors15.append('blue')


        
sizes0 = [0.0001 if c == 'blue' else 1 for c in colors0]
sizes1 = [0.0001 if c == 'blue' else 1 for c in colors1]
sizes2 = [0.0001 if c == 'blue' else 1 for c in colors2]
sizes3 = [0.0001 if c == 'blue' else 1 for c in colors3]
sizes4 = [0.0001 if c == 'blue' else 1 for c in colors4]
sizes5 = [0.0001 if c == 'blue' else 1 for c in colors5]
sizes6 = [0.0001 if c == 'blue' else 1 for c in colors6]
sizes7 = [0.0001 if c == 'blue' else 1 for c in colors7]
sizes8 = [0.0001 if c == 'blue' else 1 for c in colors8]
sizes9 = [0.0001 if c == 'blue' else 1 for c in colors9]
sizes10 = [0.0001 if c == 'blue' else 1 for c in colors10]
sizes11 = [0.0001 if c == 'blue' else 1 for c in colors11]
sizes12 = [0.0001 if c == 'blue' else 1 for c in colors12]
sizes13 = [0.0001 if c == 'blue' else 1 for c in colors13]




# plot the data using scatter with siconc values as colors
plt.figure(1)
plt.title('JUN73 TB uncorrected 22.235 GHz vs 31.4 GHz with siconc values')
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')



plt.scatter(a0chn2, a0chn1,s=sizes0, c=colors0)
plt.scatter(a1chn2, a1chn1,s=sizes1, c=colors1)
plt.scatter(a2chn2, a2chn1,s=sizes2, c=colors2)
plt.scatter(a3chn2, a3chn1,s=sizes3, c=colors3)
plt.scatter(a4chn2, a4chn1,s=sizes4, c=colors4)
plt.scatter(a5chn2, a5chn1,s=sizes5, c=colors5)
plt.scatter(a6chn2, a6chn1,s=sizes6, c=colors6)
plt.scatter(a7chn2, a7chn1,s=sizes7, c=colors7)
plt.scatter(a8chn2, a8chn1,s=sizes8, c=colors8)
plt.scatter(a9chn2, a9chn1,s=sizes9, c=colors9)
plt.scatter(a10chn2, a10chn1,s=sizes10, c=colors10)
plt.scatter(a11chn2, a11chn1,s=sizes11, c=colors11)
plt.scatter(a12chn2, a12chn1,s=sizes12, c=colors12)
plt.scatter(a13chn2, a13chn1,s=sizes13, c=colors13)


plt.xlim([100, 330])
plt.ylim([100, 330])
plt.show(1)
######


plt.figure(2)


LAT0 = ds.LAT[:, 0]
LON0 = ds.LON[:, 0]
LAT1 = ds.LAT[:, 1]
LON1 = ds.LON[:, 1]
LAT2 = ds.LAT[:, 2]
LON2 = ds.LON[:, 2]
LAT3 = ds.LAT[:, 3]
LON3 = ds.LON[:, 3]
LAT4 = ds.LAT[:, 4]
LON4 = ds.LON[:, 4]
LAT5 = ds.LAT[:, 5]
LON5 = ds.LON[:, 5]
LAT6 = ds.LAT[:, 6]
LON6 = ds.LON[:, 6]
LAT7 = ds.LAT[:, 7]
LON7 = ds.LON[:, 7]
LAT8 = ds.LAT[:, 8]
LON8 = ds.LON[:, 8]
LAT9 = ds.LAT[:, 9]
LON9 = ds.LON[:, 9]
LAT10 = ds.LAT[:, 10]
LON10 = ds.LON[:, 10]
LAT11 = ds.LAT[:, 11]
LON11 = ds.LON[:, 11]
LAT12 = ds.LAT[:, 12]
LON12 = ds.LON[:, 12]
LAT13 = ds.LAT[:, 13]
LON13 = ds.LON[:, 13]
LAT14 = ds.LAT[:, 14]
LON14 = ds.LON[:, 14]
LAT15 = ds.LAT[:, 15]
LON15 = ds.LON[:, 15]

sizesLATLON0 = [0.01 if c == 'blue' else 0.1 for c in colors0]
sizesLATLON1 = [0.01 if c == 'blue' else 0.1 for c in colors1]
sizesLATLON2 = [0.01 if c == 'blue' else 0.1 for c in colors2]
sizesLATLON3 = [0.01 if c == 'blue' else 0.1 for c in colors3]
sizesLATLON4 = [0.01 if c == 'blue' else 0.1 for c in colors4]
sizesLATLON5 = [0.01 if c == 'blue' else 0.1 for c in colors5]
sizesLATLON6 = [0.01 if c == 'blue' else 0.1 for c in colors6]
sizesLATLON7 = [0.01 if c == 'blue' else 0.1 for c in colors7]
sizesLATLON8 = [0.01 if c == 'blue' else 0.1 for c in colors8]
sizesLATLON9 = [0.01 if c == 'blue' else 0.1 for c in colors9]
sizesLATLON10 = [0.01 if c == 'blue' else 0.1 for c in colors10]
sizesLATLON11 = [0.01 if c == 'blue' else 0.1 for c in colors11]
sizesLATLON12 = [0.01 if c == 'blue' else 0.1 for c in colors12]
sizesLATLON13 = [0.01 if c == 'blue' else 0.1 for c in colors13]
sizesLATLON14 = [0.01 if c == 'blue' else 0.1 for c in colors14]
sizesLATLON15 = [0.01 if c == 'blue' else 0.1 for c in colors15]

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

plt.scatter(LON0, LAT0,s=sizesLATLON0,c=colors0)
plt.scatter(LON1, LAT1,s=sizesLATLON1,c=colors1)
plt.scatter(LON2, LAT2,s=sizesLATLON2,c=colors2)
plt.scatter(LON3, LAT3,s=sizesLATLON3,c=colors3)
plt.scatter(LON4, LAT4,s=sizesLATLON4,c=colors4)
plt.scatter(LON5, LAT5,s=sizesLATLON5,c=colors5)
plt.scatter(LON6, LAT6,s=sizesLATLON6,c=colors6)
plt.scatter(LON7, LAT7,s=sizesLATLON7,c=colors7)
plt.scatter(LON8, LAT8,s=sizesLATLON8,c=colors8)
plt.scatter(LON9, LAT9,s=sizesLATLON9,c=colors9)
plt.scatter(LON10, LAT10,s=sizesLATLON10,c=colors10)
plt.scatter(LON11, LAT11,s=sizesLATLON11,c=colors11)
plt.scatter(LON12, LAT12,s=sizesLATLON12,c=colors12)
plt.scatter(LON13, LAT13,s=sizesLATLON13,c=colors13)
plt.scatter(LON14, LAT14,s=sizesLATLON14,c=colors14)
plt.scatter(LON15, LAT15,s=sizesLATLON15,c=colors15)

plt.show(2)
