# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:35:27 2023

@author: janus
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import Normalize
import glob
import pickle

# load data:

filelist = glob.glob('C:/Users/janus/Desktop/DTU/4. Semester/30110 Fagprojekt/NEMS_colocated/*.nc')
dataset = []
for file in filelist:
    dataset.append(xr.open_dataset(file))

datajun = [dataset[20],dataset[21],dataset[22]]
datadec = [dataset[0],dataset[1],dataset[2]]

dataset=[dataset[0],dataset[1],dataset[2],dataset[20],dataset[21],dataset[22]]
#%%


decvaerdier = []

for x in range (len(datadec)):
    for i in range(len(datadec[x].siconc)):
        for j in range(14):
            lat = float(datadec[x].LAT[i,j])
            lon = float(datadec[x].LON[i,j])
            chn31 = float(datadec[x].TBNEMS[i,j,1])
            chn22 = float(datadec[x].TBNEMS[i,j,0])
            sic = float(datadec[x].siconc[i,j])
            decvaerdier.append([lat,lon,chn22,sic])
    print(x)
decvaerdier = np.array(decvaerdier)

junvaerdier = []

for x in range (len(datajun)):
    for i in range(len(datajun[x].siconc)):
        for j in range(14):
            lat = float(datajun[x].LAT[i,j])
            lon = float(datajun[x].LON[i,j])
            chn31 = float(datajun[x].TBNEMS[i,j,1])
            chn22 = float(datajun[x].TBNEMS[i,j,0])
            sic = float(datajun[x].siconc[i,j])
            junvaerdier.append([lat,lon,chn22,sic])
    print(x)
junvaerdier = np.array(junvaerdier)

with open('raadatadeclatlonchn1sic.pkl', 'wb') as f:
    pickle.dump(decvaerdier, f)
with open('raadatajunlatlonchn1sic.pkl', 'wb') as f:
    pickle.dump(junvaerdier, f)

#%%

meandatasetnordis1 = []
meandatasetnordis2 = []
meandatasetnordvand1 = []
meandatasetnordvand2 = []
meandatasetsydis1 = []
meandatasetsydis2 = []
meandatasetsydvand1 = []
meandatasetsydvand2 = []

stdnordis1 = []
stdnordis2 = []
stdnordvand1 = []
stdnordvand2 = []
stdsydis1 = []
stdsydis2 = []
stdsydvand1 = []
stdsydvand2 = []


altkolddatanord = []
altkolddatasyd = []
for x in range(len(dataset)):
    nydatasydis1 = []
    nydatasydis2 = []
    nydatasydvand1 = []
    nydatasydvand2 = []
    nydatanordis1 = []
    nydatanordis2 = []
    nydatanordvand1 = []
    nydatanordvand2 = []
    
    for i in range(len(dataset[x].sst[:,0])):
        for j in range(14):
            #Vi frasorterer data med en surface temperatur over 5 grader celcius, da vi kigger på koldt vand
            if dataset[x].sst[i,j]<278:
                #Tager data for sydlige halvkugle
                if dataset[x].LAT[i,j]<0:
                    altkolddatanord.append([x,i,j])
                    #Tager data som klassificeres som is (siconc>0.9) og med surface temperature under 0 celcius
                    if dataset[x].siconc[i,j]>0.9 and dataset[x].sst[i,j]<273.15:
                        if dataset[x].TBNEMS[i,j,0]<273.15 and dataset[x].TBNEMS[i,j,0]>100:
                            nydatasydis1.append(dataset[x].TBNEMS[i,j,0])
                        if dataset[x].TBNEMS[i,j,1]<273.15 and dataset[x].TBNEMS[i,j,1]>100:
                            nydatasydis2.append(dataset[x].TBNEMS[i,j,1])
                    #Tager data som klassificeres som vand (siconc=0)
                    elif dataset[x].siconc[i,j]==0:
                        if dataset[x].TBNEMS[i,j,0]<273.15 and dataset[x].TBNEMS[i,j,0]>100:
                            nydatasydvand1.append(dataset[x].TBNEMS[i,j,0])
                        if dataset[x].TBNEMS[i,j,1]<273.15 and dataset[x].TBNEMS[i,j,1]>100:
                            nydatasydvand2.append(dataset[x].TBNEMS[i,j,1])
                #Tager data for nordlige halvkugle halvkugle
                elif dataset[x].LAT[i,j]>0:
                    altkolddatasyd.append([x,i,j])
                    if dataset[x].siconc[i,j]>0.9 and dataset[x].sst[i,j]<273.15:
                        if dataset[x].TBNEMS[i,j,0]<273.15 and dataset[x].TBNEMS[i,j,0]>100:
                            nydatanordis1.append(dataset[x].TBNEMS[i,j,0])
                        if dataset[x].TBNEMS[i,j,1]<273.15 and dataset[x].TBNEMS[i,j,1]>100:
                            nydatanordis2.append(dataset[x].TBNEMS[i,j,1])
                    elif dataset[x].siconc[i,j]==0:
                        if dataset[x].TBNEMS[i,j,0]<273.15 and dataset[x].TBNEMS[i,j,0]>100:
                            nydatanordvand1.append(dataset[x].TBNEMS[i,j,0])
                        if dataset[x].TBNEMS[i,j,1]<273.15 and dataset[x].TBNEMS[i,j,1]>100:
                            nydatanordvand2.append(dataset[x].TBNEMS[i,j,1])
    print(x)
    
    
    stdnordis1.append(np.std(nydatanordis1))
    stdnordis2.append(np.std(nydatanordis2))
    stdnordvand1.append(np.std(nydatanordvand1))
    stdnordvand2.append(np.std(nydatanordvand2))
    stdsydis1.append(np.std(nydatasydis1))
    stdsydis2.append(np.std(nydatasydis2))
    stdsydvand1.append(np.std(nydatasydvand1))
    stdsydvand2.append(np.std(nydatasydvand2))
    
    meandatasetnordis1.append(np.mean(nydatanordis1))
    meandatasetnordis2.append(np.mean(nydatanordis2))
    meandatasetnordvand1.append(np.mean(nydatanordvand1))
    meandatasetnordvand2.append(np.mean(nydatanordvand2))
    meandatasetsydis1.append(np.mean(nydatasydis1))
    meandatasetsydis2.append(np.mean(nydatasydis2))
    meandatasetsydvand1.append(np.mean(nydatasydvand1))
    meandatasetsydvand2.append(np.mean(nydatasydvand2))
    print(f"For loop progress dataset :{x}")

print(f" Nordis1 :{meandatasetnordis1}")
print(f" Nordis2 :{meandatasetnordis2}")
print(f" Sydis1 :{meandatasetsydis1}")
print(f" Sydis2:{meandatasetsydis2}")
print(f" Nordvand1:{meandatasetnordvand1}")
print(f" Nordvand2:{meandatasetnordvand2}")
print(f" Sydvand1:{meandatasetsydvand1}")
print(f" Sydvand2:{meandatasetsydvand2}")