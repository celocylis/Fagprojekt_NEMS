# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:45:19 2023

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

#Import fy og my

with open('tp1nfirstyear.pkl', 'rb') as f:
    tp1nfirstyear = pickle.load(f)

with open('tp1sfirstyear.pkl', 'rb') as f:
    tp1sfirstyear = pickle.load(f)

with open('tp1nmultiyear.pkl', 'rb') as f:
    tp1nmultiyear = pickle.load(f)

with open('tp1smultiyear.pkl', 'rb') as f:
    tp1smultiyear = pickle.load(f)  

with open('tp2nfirstyear.pkl', 'rb') as f:
    tp2nfirstyear = pickle.load(f)  

with open('tp2sfirstyear.pkl', 'rb') as f:
    tp2sfirstyear = pickle.load(f)  

with open('tp2nmultiyear.pkl', 'rb') as f:
    tp2nmultiyear = pickle.load(f)  

with open('tp2smultiyear.pkl', 'rb') as f:
    tp2smultiyear = pickle.load(f)  

with open('tpnv1.pkl', 'rb') as f:
    tpnv1 = pickle.load(f)

with open('tpnv2.pkl', 'rb') as f:
    tpnv2 = pickle.load(f)

with open('tpsv1.pkl', 'rb') as f:
    tpsv1 = pickle.load(f)

with open('tpsv2.pkl', 'rb') as f:
    tpsv2 = pickle.load(f) 


#NORD
#åbent vand              
tpvand22n = tpnv1
tpvand31n = tpnv2
#first year ice
tpfy22n = tp1nfirstyear
tpfy31n = tp2nfirstyear
#multiyear ice
tpmy22n = tp1nmultiyear
tpmy31n = tp2nmultiyear

#SYD
#åbent vand              
tpvand22s = tpsv1
tpvand31s = tpsv2
#first year ice
tpfy22s = tp1sfirstyear
tpfy31s = tp2sfirstyear
#multiyear ice
tpmy22s = tp1smultiyear
tpmy31s = tp2smultiyear

fcomisodec72 = []
fcomisojun73 = []

progcheck = 0
for x in range(len(dec72)):
    progcheck = progcheck+1
    fcomisotemp = []
    for i in range(len(dec72[x].TBNEMS)):
        for j in range(14):
            lat = float(dec72[x].LAT[i,j])
            lon = float(dec72[x].LON[i,j])
            TB22 = float(dec72[x].TBNEMS[i,j,0])
            TB31 = float(dec72[x].TBNEMS[i,j,1])
            if lat > 50:
                af = (tpfy31n[x] - tpmy31n[x])/(tpfy22n[x] - tpmy22n[x])
                bf = (tpmy31n[x] - af*tpmy22n[x])
                qf = (TB31 - tpvand31n[x])/(TB22 - tpvand22n[x])
                wf = (tpvand31n[x] - qf*tpvand22n[x])
                ti18vf = (bf - wf)/(qf - af)
                siconccalc = (TB22 - tpvand22n[x])/(ti18vf - tpvand22n[x])
                fcomisotemp.append([siconccalc,lat,lon])
            elif lat < -50:
                af = (tpfy31s[x] - tpmy31s[x])/(tpfy22s[x] - tpmy22s[x])
                bf = (tpmy31s[x] - af*tpmy22s[x])
                qf = (TB31 - tpvand31s[x])/(TB22 - tpvand22s[x])
                wf = (tpvand31s[x] - qf*tpvand22s[x])
                ti18vf = (bf - wf)/(qf - af)
                siconccalc = (TB22 - tpvand22s[x])/(ti18vf - tpvand22s[x])
                fcomisotemp.append([siconccalc,lat,lon])
    print(f"progcheck dec :{progcheck} dataset er done")
    fcomisodec72.append(np.array(fcomisotemp))

progcheck = 0

for x in range(len(jun73)):
    progcheck = progcheck+1
    fcomisotemp = []
    for i in range(len(jun73[x].TBNEMS)):
        for j in range(14):
            lat = float(jun73[x].LAT[i,j])
            lon = float(jun73[x].LON[i,j])
            TB22 = float(jun73[x].TBNEMS[i,j,0])
            TB31 = float(jun73[x].TBNEMS[i,j,1])
            if lat > 50:
                af = (tpfy31n[x] - tpmy31n[x])/(tpfy22n[x] - tpmy22n[x])
                bf = (tpmy31n[x] - af*tpmy22n[x])
                qf = (TB31 - tpvand31n[x])/(TB22 - tpvand22n[x])
                wf = (tpvand31n[x] - qf*tpvand22n[x])
                ti18vf = (bf - wf)/(qf - af)
                siconccalc = (TB22 - tpvand22n[x])/(ti18vf - tpvand22n[x])
                fcomisotemp.append([siconccalc,lat,lon])
            elif lat < -50:
                af = (tpfy31s[x] - tpmy31s[x])/(tpfy22s[x] - tpmy22s[x])
                bf = (tpmy31s[x] - af*tpmy22s[x])
                qf = (TB31 - tpvand31s[x])/(TB22 - tpvand22s[x])
                wf = (tpvand31s[x] - qf*tpvand22s[x])
                ti18vf = (bf - wf)/(qf - af)
                siconccalc = (TB22 - tpvand22s[x])/(ti18vf - tpvand22s[x])
                fcomisotemp.append([siconccalc,lat,lon])
    print(f"progcheck jun :{progcheck} dataset er done")
    fcomisojun73.append(np.array(fcomisotemp))

combined_listdec = []
combined_listdec.extend(fcomisodec72[0])
combined_listdec.extend(fcomisodec72[1])
combined_listdec.extend(fcomisodec72[2])
combined_listdec=np.array(combined_listdec)

combined_listjun = []
combined_listjun.extend(fcomisojun73[0])
combined_listjun.extend(fcomisojun73[1])
combined_listjun.extend(fcomisojun73[2])
combined_listjun=np.array(combined_listjun)

with open('testafcomisodec.pkl', 'wb') as f:
    pickle.dump(combined_listdec, f)
with open('testafcomisojun.pkl', 'wb') as f:
    pickle.dump(combined_listjun, f)
