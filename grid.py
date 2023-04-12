# -*- coding: utf-8 -*-
"""
@author: noahe 
"""

import numpy as np
import matplotlib.pyplot as plt
import pyresample as pr
import numpy as np
import xarray as xr


# load data:
ds = xr.open_dataset('C:/Users/noahe/OneDrive/Dokumenter/DTU/Fagprojekt/NEMS_colocated/Nimbus5-NEMS_L2_1972m1217t011144_DR24_era5.nc')
#ds = xr.open_dataset('C:/Users/noahe/OneDrive/Dokumenter/DTU/Fagprojekt/NEMS_colocated/Nimbus5-NEMS_L2_1973m0617t031041_DR26_era5.nc')

colors_list = []
GR=[]

area_def_north_ease = pr.utils.parse_area_file('./areas_ease.cfg', 'ease_nh')[0]
area_def_south_ease = pr.utils.parse_area_file('./areas_ease.cfg', 'ease_sh')[0]

for i in range(14):
    colors = []
    GR = (ds.TBNEMS[:,i,1]-ds.TBNEMS[:,i,0])/(ds.TBNEMS[:,i,0]+ds.TBNEMS[:,i,1]) 
    
    for value in GR:
        if value > 0.02:
            colors.append('blue')
        elif value <= -0.02:
            colors.append('black')
        else:
            colors.append('green')
    colors_list.append(colors)
    
plt.figure(1,figsize=(24,8));

for i in range(16):
    globals()[f'LAT{i}'] = ds.LAT[:, i]
    globals()[f'LON{i}'] = ds.LON[:, i]
    globals()[f'sst{i}'] = ds.sst[:, i]

#ice is only in cold water, if the water is 5 degrees hot there is probably no ice in it
ice_infested_waters = (globals()[f'sst{i}'] <= 278.0)
open_water = (globals()[f'sst{i}'] > 278.0)
#geografisk udvaelgelse, her var det før lat > 0 og lat < 0, der er ingen is syd for 78.40 deg.
# geographical selection, here it was before lat> 0 and lat <0, there is no ice south of 78.40 deg.
arctic = (globals()[f'LAT{i}'] > 32)
north_pole=(globals()[f'LAT{i}'] < 90)
antarctic =(globals()[f'LAT{i}'] < -48)
south_pole=(globals()[f'LAT{i}'] > -90)
ocean=(globals()[f'sst{i}'] == 0)
non_ocean = (globals()[f'sst{i}']>0)

arctic_ice = arctic & north_pole
antarctic_ice = antarctic & south_pole & ice_infested_waters & non_ocean

#resample the grids
#lon_n, lat_n = pr.utils.check_and_wrap(LON[i][arctic_ice], LAT[i][arctic_ice])
#lon_n, lat_n = pr.utils.check_and_wrap(LON[i][antarctic_ice], LAT[i][antarctic_ice])
lon_n, lat_n = pr.utils.check_and_wrap(globals()[f'LON{i}'][arctic_ice], globals()[f'LAT{i}'] = ds.LAT[:, i][arctic_ice])
lon_s, lat_s = pr.utils.check_and_wrap(globals()[f'LON{i}'][antarctic_ice], globals()[f'LAT{i}'] = ds.LAT[:, i][antarctic_ice])
  

  
#swath_def = pr.geometry.SwathDefinition(lons=lon, lats=lat)  
arctic_def = pr.geometry.SwathDefinition(lons=lon_n, lats=lat_n)
antarctic_def = pr.geometry.SwathDefinition(lons=lon_s, lats=lat_s)

gr =   np.concatenate(GR)

n_GR = pr.kd_tree.resample_nearest(arctic_def, gr,\
                                   area_def_north_ease, radius_of_influence=50000,fill_value=None) 
plt.imshow(n_GR,vmin=0,vmax=1)
plt.colorbar()
plt.title('GR for nordpolen i december')
plt.show()
