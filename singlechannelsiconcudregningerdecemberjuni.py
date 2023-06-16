# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:38:03 2023

@author: janus
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  2 08:56:55 2023

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

#Definerer forskellige måneders data
dec72=[dataset[0],dataset[1],dataset[2]]
jun73=[dataset[20],dataset[21],dataset[22]]


tpnordis1=np.loadtxt('meandatasetnordis1.txt', delimiter=',', skiprows=0)
tpnordis2=np.loadtxt('meandatasetnordis2.txt', delimiter=',', skiprows=0)
tpnordvand1=np.loadtxt('meandatasetnordvand1.txt', delimiter=',', skiprows=0)
tpnordvand2=np.loadtxt('meandatasetnordvand2.txt', delimiter=',', skiprows=0)
tpsydis1=np.loadtxt('meandatasetsydis1.txt', delimiter=',', skiprows=0)
tpsydis2=np.loadtxt('meandatasetsydis2.txt', delimiter=',', skiprows=0)
tpsydvand1=np.loadtxt('meandatasetsydvand1.txt', delimiter=',', skiprows=0)
tpsydvand2=np.loadtxt('meandatasetsydvand2.txt', delimiter=',', skiprows=0)

#Laver nye tiepoints lister med hver måned istedet for hver dataset
tpnordis1ny=[tpnordis1[0:3],tpnordis1[20:23]]
tpnordis2ny=[tpnordis2[0:3],tpnordis2[20:23]]
tpnordvand1ny=[tpnordvand1[0:3],tpnordvand1[20:23]]
tpnordvand2ny=[tpnordvand2[0:3],tpnordvand2[20:23]]
tpsydis1ny=[tpsydis1[0:3],tpsydis1[20:23]]
tpsydis2ny=[tpsydis2[0:3],tpsydis2[20:23]]
tpsydvand1ny=[tpsydvand1[0:3],tpsydvand1[20:23]]
tpsydvand2ny=[tpsydvand2[0:3],tpsydvand2[20:23]]


datamaaned = [dec72,jun73]


Siconc1nhvermaaned = []
Siconc1shvermaaned = []
Siconc2nhvermaaned = []
Siconc2shvermaaned = []
progcheck = 0

Siconc1nhverset = []
Siconc1shverset = []
Siconc2nhverset = []
Siconc2shverset = []

for x in range(len(datamaaned)):
    sico1n = []
    sico1s = []
    sico2n = []
    sico2s = []
    for u in range(len(datamaaned[x])):
        sico1nx = []
        sico1sx = []
        sico2nx = []
        sico2sx = []
        progcheck = progcheck+1
        for i in range(len(datamaaned[x][u].TBNEMS)):
            for j in range(14):
                if datamaaned[x][u].sst[i,j]<278 and datamaaned[x][u].TBNEMS[i,j,0]>100:
                    lat = float(datamaaned[x][u].LAT[i,j])
                    lon = float(datamaaned[x][u].LON[i,j])
                    if lat > 50:
                        chn1 = float(datamaaned[x][u].TBNEMS[i,j,0])
                        chn2 = float(datamaaned[x][u].TBNEMS[i,j,1])
                        sic1 = (chn1-tpnordvand1ny[x][u])/(tpnordis1ny[x][u]-tpnordvand1ny[x][u])
                        sic2 = (chn2-tpnordvand2ny[x][u])/(tpnordis2ny[x][u]-tpnordvand2ny[x][u])
                        sico1n.append([lat,lon,sic1,chn1])
                        sico2n.append([lat,lon,sic2,chn2])
                        sico1nx.append([lat,lon,sic1,chn1])
                        sico2nx.append([lat,lon,sic2,chn2])
                    elif lat < -50:
                        chn1 = float(datamaaned[x][u].TBNEMS[i,j,0])
                        chn2 = float(datamaaned[x][u].TBNEMS[i,j,1])
                        sic1 = (chn1-tpsydvand1ny[x][u])/(tpsydis1ny[x][u]-tpsydvand1ny[x][u])
                        sic2 = (chn2-tpsydvand2ny[x][u])/(tpsydis2ny[x][u]-tpsydvand2ny[x][u])
                        sico1s.append([lat,lon,sic1,chn1])
                        sico2s.append([lat,lon,sic2,chn2])    
                        sico1sx.append([lat,lon,sic1,chn1])
                        sico2sx.append([lat,lon,sic2,chn2]) 
        Siconc1nhverset.append(sico1nx)
        Siconc1shverset.append(sico1sx)
        Siconc2nhverset.append(sico2nx)
        Siconc2shverset.append(sico2sx)
        print(f"progcheck :{progcheck} dataset er done")
    Siconc1nhvermaaned.append(sico1n)
    Siconc1shvermaaned.append(sico1s)
    Siconc2nhvermaaned.append(sico2n)
    Siconc2shvermaaned.append(sico2s)
    print(f"For loop progress datamaaned :{x}")
    

#Gem data i god fil
with open('nordsiconcchn1decjun.pkl', 'wb') as f:
    pickle.dump(Siconc1nhvermaaned, f)

with open('nordsiconcchn2decjun.pkl', 'wb') as f:
    pickle.dump(Siconc2nhvermaaned, f)

with open('sydsiconcchn1decjun.pkl', 'wb') as f:
    pickle.dump(Siconc1shvermaaned, f)

with open('sydsiconcchn2decjun.pkl', 'wb') as f:
    pickle.dump(Siconc2shvermaaned, f)
    
with open('nordsiconcchn1hverset.pkl', 'wb') as f:
    pickle.dump(Siconc1nhverset, f)

with open('nordsiconcchn2hverset.pkl', 'wb') as f:
    pickle.dump(Siconc2nhverset, f)

with open('sydsiconcchn1hverset.pkl', 'wb') as f:
    pickle.dump(Siconc1shverset, f)

with open('sydsiconcchn2hverset.pkl', 'wb') as f:
    pickle.dump(Siconc2shverset, f)
    
tpnv1 = [tpnordvand1[0],tpnordvand1[1],tpnordvand1[2],tpnordvand1[20],tpnordvand1[21],tpnordvand1[22]]
tpnv2 = [tpnordvand2[0],tpnordvand2[1],tpnordvand2[2],tpnordvand2[20],tpnordvand2[21],tpnordvand2[22]]
tpsv1 = [tpsydvand1[0],tpsydvand1[1],tpsydvand1[2],tpsydvand1[20],tpsydvand1[21],tpsydvand1[22]] 
tpsv2 = [tpsydvand2[0],tpsydvand2[1],tpsydvand2[2],tpsydvand2[20],tpsydvand2[21],tpsydvand2[22]] 

with open('tpnv1.pkl', 'wb') as f:
    pickle.dump(tpnv1, f)

with open('tpnv2.pkl', 'wb') as f:
    pickle.dump(tpnv2, f)

with open('tpsv1.pkl', 'wb') as f:
    pickle.dump(tpsv1, f)

with open('tpsv2.pkl', 'wb') as f:
    pickle.dump(tpsv2, f)