# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:36:42 2023

@author: janus
"""
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from matplotlib.colors import Normalize
import glob
import pickle

#[nyT31,nyT22,lat,lon,sst,SIC,era5sic]

with open('RTMkorrigeretDecember.pkl', 'rb') as f:
    RTMDecember = pickle.load(f)

with open('RTMKorrigeretJuni.pkl', 'rb') as f:
    RTMJuni = pickle.load(f)


#%% Tiepoints beregning
decnordis1 = []
decnordis2 = []
decnordvand1 = []
decnordvand2 = []
decsydis1 = []
decsydis2 = []
decsydvand1 = []
decsydvand2 = []

junnordis1 = []
junnordis2 = []
junnordvand1 = []
junnordvand2 = []
junsydis1 = []
junsydis2 = []
junsydvand1 = []
junsydvand2 = []

#[2=31ghz, 1=22ghz]

for x in range (len(RTMDecember)):
    if RTMDecember[x,4] < 278:
        if RTMDecember[x,2] > 0:
            if RTMDecember[x,5] > 0.9:
                if RTMDecember[x,1] > 100:
                    decnordis1.append(RTMDecember[x,1])
                if RTMDecember[x,0] > 100:
                    decnordis2.append(RTMDecember[x,0])
            elif RTMDecember[x,5] < 0.3:
                if RTMDecember[x,1] > 100:
                    decnordvand1.append(RTMDecember[x,1])
                if RTMDecember[x,0] > 100:
                    decnordvand2.append(RTMDecember[x,0])
        elif RTMDecember[x,2] < 0:
            if RTMDecember[x,5] > 0.9:
                if RTMDecember[x,1] > 100:
                    decsydis1.append(RTMDecember[x,1])
                if RTMDecember[x,0] > 100:
                    decsydis2.append(RTMDecember[x,0])
            elif RTMDecember[x,5] < 0.3:
                if RTMDecember[x,1] > 100:
                    decsydvand1.append(RTMDecember[x,1])
                if RTMDecember[x,0] > 100:
                    decsydvand2.append(RTMDecember[x,0])
                
for x in range (len(RTMJuni)):
    if RTMJuni[x,4] < 278:
        if RTMJuni[x,2] > 0:
            if RTMJuni[x,5] > 0.9:
                if RTMJuni[x,1] > 100:
                    junnordis1.append(RTMJuni[x,1])
                if RTMJuni[x,0] > 100:
                    junnordis2.append(RTMJuni[x,0])
            elif RTMJuni[x,5] < 0.3:
                if RTMJuni[x,1] > 100:
                    junnordvand1.append(RTMJuni[x,1])
                if RTMJuni[x,0] > 100:
                    junnordvand2.append(RTMJuni[x,0])
        elif RTMJuni[x,2] < 0:
            if RTMJuni[x,5] > 0.9:
                if RTMJuni[x,1] > 100:
                    junsydis1.append(RTMJuni[x,1])
                if RTMJuni[x,0] > 100:
                    junsydis2.append(RTMJuni[x,0])
            elif RTMJuni[x,5] < 0.3:
                if RTMJuni[x,1] > 100:
                    junsydvand1.append(RTMJuni[x,1])
                if RTMJuni[x,0] > 100:
                    junsydvand2.append(RTMJuni[x,0])

stddecnordvand1 = np.std(decnordvand1)
stddecnordvand2 = np.std(decnordvand2)
stddecsydvand1 = np.std(decsydvand1)
stddecsydvand2 = np.std(decsydvand2)
stddecnordis1 = np.std(decnordis1)
stddecnordis2 = np.std(decnordis2)
stddecsydis1 = np.std(decsydis1)
stddecsydis2 = np.std(decsydis2)

stdjunnordvand1 = np.std(junnordvand1)
stdjunnordvand2 = np.std(junnordvand2)
stdjunsydvand1 = np.std(junsydvand1)
stdjunsydvand2 = np.std(junsydvand2)
stdjunnordis1 = np.std(junnordis1)
stdjunnordis2 = np.std(junnordis2)
stdjunsydis1 = np.std(junsydis1)
stdjunsydis2 = np.std(junsydis2)

TPdecnordis1 = np.mean(decnordis1)
TPdecnordis2 = np.mean(decnordis2)
TPdecnordvand1 = np.mean(decnordvand1)
TPdecnordvand2 = np.mean(decnordvand2)
TPdecsydis1 = np.mean(decsydis1)
TPdecsydis2 = np.mean(decsydis2)
TPdecsydvand1 = np.mean(decsydvand1)
TPdecsydvand2 = np.mean(decsydvand2)

TPjunnordis1 = np.mean(junnordis1)
TPjunnordis2 = np.mean(junnordis2)
TPjunnordvand1 = np.mean(junnordvand1)
TPjunnordvand2 = np.mean(junnordvand2)
TPjunsydis1 = np.mean(junsydis1)
TPjunsydis2 = np.mean(junsydis2)
TPjunsydvand1 = np.mean(junsydvand1)
TPjunsydvand2 = np.mean(junsydvand2)

#%% 1chn siconc
#[nyT31,nyT22,lat,lon,sst,SIC,era5sic]
#[2=31ghz, 1=22ghz]

DecSiconc31n = []
DecSiconc31s = []
DecSiconc22n = []
DecSiconc22s = []

for x in range(len(RTMDecember)):
    lat = RTMDecember[x,2]
    lon = RTMDecember[x,3]
    T31 = RTMDecember[x,0]
    T22 = RTMDecember[x,1]
    
    if RTMDecember[x,4] < 278:
        if RTMDecember[x,2] > 0:
            if RTMDecember[x,0] > 100:
                sic1 = (T31-TPdecnordvand2)/(TPdecnordis2-TPdecnordvand2)            
                DecSiconc31n.append([lat,lon,sic1,T31,T22])
            if RTMDecember[x,1] > 100:
                sic2 = (T31-TPdecnordvand1)/(TPdecnordis1-TPdecnordvand1)  
                DecSiconc22n.append([lat,lon,sic2,T31,T22])
        elif RTMDecember[x,2] < 0:
            if RTMDecember[x,0] > 100:
                sic1 = (T31-TPdecsydvand2)/(TPdecsydis2-TPdecsydvand2)            
                DecSiconc31s.append([lat,lon,sic1,T31,T22])
            if RTMDecember[x,1] > 100:
                sic2 = (T31-TPdecsydvand1)/(TPdecsydis1-TPdecsydvand1)  
                DecSiconc22s.append([lat,lon,sic2,T31,T22])

JunSiconc31n = []
JunSiconc31s = []
JunSiconc22n = []
JunSiconc22s = []

for x in range(len(RTMJuni)):
    lat = RTMJuni[x,2]
    lon = RTMJuni[x,3]
    T31 = RTMJuni[x,0]
    T22 = RTMJuni[x,1]
    
    if RTMJuni[x,4] < 278:
        if RTMJuni[x,2] > 0:
            if RTMJuni[x,0] > 100:
                sic1 = (T31-TPjunnordvand2)/(TPjunnordis2-TPjunnordvand2)            
                JunSiconc31n.append([lat,lon,sic1,T31,T22])
            if RTMJuni[x,1] > 100:
                sic2 = (T31-TPjunnordvand1)/(TPjunnordis1-TPjunnordvand1)  
                JunSiconc22n.append([lat,lon,sic2,T31,T22])
        elif RTMJuni[x,2] < 0:
            if RTMJuni[x,0] > 100:
                sic1 = (T31-TPjunsydvand2)/(TPjunsydis2-TPjunsydvand2)            
                JunSiconc31s.append([lat,lon,sic1,T31,T22])
            if RTMJuni[x,1] > 100:
                sic2 = (T31-TPjunsydvand1)/(TPjunsydis1-TPjunsydvand1)  
                JunSiconc22s.append([lat,lon,sic2,T31,T22])

DecSiconc31n = np.array(DecSiconc31n)
DecSiconc22n = np.array(DecSiconc22n)
DecSiconc31s = np.array(DecSiconc31s)
DecSiconc22s = np.array(DecSiconc22s)
JunSiconc31n = np.array(JunSiconc31n)
JunSiconc22n = np.array(JunSiconc22n)
JunSiconc31s = np.array(JunSiconc31s)
JunSiconc22s = np.array(JunSiconc22s)

with open('RTM1chnDec31N.pkl', 'wb') as f:
    pickle.dump(DecSiconc31n, f)
with open('RTM1chnDec31S.pkl', 'wb') as f:
    pickle.dump(DecSiconc31s, f)
with open('RTM1chnDec22N.pkl', 'wb') as f:
    pickle.dump(DecSiconc22n, f)
with open('RTM1chnDec22S.pkl', 'wb') as f:
    pickle.dump(DecSiconc22s, f)

with open('RTM1chnJun31N.pkl', 'wb') as f:
    pickle.dump(JunSiconc31n, f)
with open('RTM1chnJun31S.pkl', 'wb') as f:
    pickle.dump(JunSiconc31s, f)
with open('RTM1chnJun22N.pkl', 'wb') as f:
    pickle.dump(JunSiconc22n, f)
with open('RTM1chnJun22S.pkl', 'wb') as f:
    pickle.dump(JunSiconc22s, f)

#%% Firstyear og Multiyear tiepoints

filterDecSiconc31n = DecSiconc31n[DecSiconc31n[:,2]>0.9]
filterDecSiconc31s = DecSiconc31s[DecSiconc31s[:,2]>0.9]
filterDecSiconc22n = DecSiconc22n[DecSiconc22n[:,2]>0.9]
filterDecSiconc22s = DecSiconc22s[DecSiconc22s[:,2]>0.9]
filterJunSiconc31n = JunSiconc31n[JunSiconc31n[:,2]>0.9]
filterJunSiconc31s = JunSiconc31s[JunSiconc31s[:,2]>0.9]
filterJunSiconc22n = JunSiconc22n[JunSiconc22n[:,2]>0.9]
filterJunSiconc22s = JunSiconc22s[JunSiconc22s[:,2]>0.9]

sortDecSiconc31n = filterDecSiconc31n[np.argsort(filterDecSiconc31n[:,2])]
sortDecSiconc31s = filterDecSiconc31s[np.argsort(filterDecSiconc31s[:,2])]
sortDecSiconc22n = filterDecSiconc22n[np.argsort(filterDecSiconc22n[:,2])]
sortDecSiconc22s = filterDecSiconc22s[np.argsort(filterDecSiconc22s[:,2])]
sortJunSiconc31n = filterJunSiconc31n[np.argsort(filterJunSiconc31n[:,2])]
sortJunSiconc31s = filterJunSiconc31s[np.argsort(filterJunSiconc31s[:,2])]
sortJunSiconc22n = filterJunSiconc22n[np.argsort(filterJunSiconc22n[:,2])]
sortJunSiconc22s = filterJunSiconc22s[np.argsort(filterJunSiconc22s[:,2])]

antalraekkerDec31n=int(len(sortDecSiconc31n)*0.2)
stdmyDec31n = np.std(sortDecSiconc31n[:antalraekkerDec31n,3])
myDec31n=np.mean(sortDecSiconc31n[:antalraekkerDec31n,3])
stdfyDec31n = np.std(sortDecSiconc31n[-antalraekkerDec31n:,3])
fyDec31n=np.mean(sortDecSiconc31n[-antalraekkerDec31n:,3])

antalraekkerDec31s=int(len(sortDecSiconc31s)*0.2)
stdmyDec31s = np.std(sortDecSiconc31s[:antalraekkerDec31s,3])
myDec31s=np.mean(sortDecSiconc31s[:antalraekkerDec31s,3])
stdfyDec31s = np.std(sortDecSiconc31s[-antalraekkerDec31s:,3])
fyDec31s=np.mean(sortDecSiconc31s[-antalraekkerDec31s:,3])

antalraekkerDec22n=int(len(sortDecSiconc22n)*0.2)
stdmyDec22n = np.std(sortDecSiconc22n[:antalraekkerDec22n,4])
myDec22n=np.mean(sortDecSiconc22n[:antalraekkerDec22n,4])
stdfyDec22n = np.std(sortDecSiconc22n[-antalraekkerDec22n:,4])
fyDec22n=np.mean(sortDecSiconc22n[-antalraekkerDec22n:,4])

antalraekkerDec22s=int(len(sortDecSiconc22s)*0.2)
stdmyDec22s = np.std(sortDecSiconc22s[:antalraekkerDec22s,4])
myDec22s=np.mean(sortDecSiconc22s[:antalraekkerDec22s,4])
stdfyDec22s = np.std(sortDecSiconc22s[-antalraekkerDec22s:,4])
fyDec22s=np.mean(sortDecSiconc22s[-antalraekkerDec22s:,4])

antalraekkerJun31n=int(len(sortJunSiconc31n)*0.2)
stdmyJun31n = np.std(sortJunSiconc31n[:antalraekkerJun31n,3])
myJun31n=np.mean(sortJunSiconc31n[:antalraekkerJun31n,3])
stdfyJun31n = np.std(sortJunSiconc31n[-antalraekkerJun31n:,3])
fyJun31n=np.mean(sortJunSiconc31n[-antalraekkerJun31n:,3])

antalraekkerJun31s=int(len(sortJunSiconc31s)*0.2)
stdmyJun31s = np.std(sortJunSiconc31s[:antalraekkerJun31s,3])
myJun31s=np.mean(sortJunSiconc31s[:antalraekkerJun31s,3])
stdfyJun31s = np.std(sortJunSiconc31s[-antalraekkerJun31s:,3])
fyJun31s=np.mean(sortJunSiconc31s[-antalraekkerJun31s:,3])

antalraekkerJun22n=int(len(sortJunSiconc22n)*0.2)
stdmyJun22n = np.std(sortJunSiconc22n[:antalraekkerJun22n,4])
myJun22n=np.mean(sortJunSiconc22n[:antalraekkerJun22n,4])
stdfyJun22n = np.std(sortJunSiconc22n[-antalraekkerJun22n:,4])
fyJun22n=np.mean(sortJunSiconc22n[-antalraekkerJun22n:,4])

antalraekkerJun22s=int(len(sortJunSiconc22s)*0.2)
stdmyJun22s = np.std(sortJunSiconc22s[:antalraekkerJun22s,4])
myJun22s=np.mean(sortJunSiconc22s[:antalraekkerJun22s,4])
stdfyJun22s = np.std(sortJunSiconc22s[-antalraekkerJun22s:,4])
fyJun22s=np.mean(sortJunSiconc22s[-antalraekkerJun22s:,4])



plt.scatter(fyDec31n,fyDec22n,color='red')
plt.scatter(myDec31n,myDec22n,color='black')
plt.scatter(TPdecnordvand2,TPdecnordvand1,color='blue')
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')
plt.title("December 72 Tiepoints Nord")
plt.xlim([140, 260])
plt.ylim([140, 260])
plt.show(1)

plt.scatter(fyJun31n,fyJun22n,color='red')
plt.scatter(myJun31n,myJun22n,color='black')
plt.scatter(TPjunnordvand2,TPjunnordvand1,color='blue')
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')
plt.title("Juni 73 Tiepoints Nord")
plt.xlim([140, 260])
plt.ylim([140, 260])
plt.show(2)


plt.scatter(fyDec31s,fyDec22s,color='red')
plt.scatter(myDec31s,myDec22s,color='black')
plt.scatter(TPdecsydvand2,TPdecsydvand1,color='blue')
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')
plt.title("December 72 Tiepoints Syd")
plt.xlim([140, 260])
plt.ylim([140, 260])
plt.show(3)

plt.scatter(fyJun31s,fyJun22s,color='red')
plt.scatter(myJun31s,myJun22s,color='black')
plt.scatter(TPjunsydvand2,TPjunsydvand1,color='blue')
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')
plt.title("Juni 73 Tiepoints Syd")
plt.xlim([140, 260])
plt.ylim([140, 260])
plt.show(4)

#%%FCOMISO
#Format for RTMDecember og RTMJuni [nyT31,nyT22,lat,lon,sst,SIC,era5sic]

RTMfcomisoDec = []
RTMfcomisoJun = []

for x in range(len(RTMDecember)):
    lat = RTMDecember[x,2]
    lon = RTMDecember[x,3]
    T31 = RTMDecember[x,0]
    T22 = RTMDecember[x,1]
    sst = RTMDecember[x,4]
    if T31 > 100 and T22 > 100 and sst < 278:
        if lat > 0:
            af = (fyDec31n - myDec31n)/(fyDec22n - myDec22n)
            bf = (myDec31n - af*myDec22n)
            qf = (T31 - TPdecnordvand2)/(T22 - TPdecnordvand1)
            wf = (TPdecnordvand2 - qf*TPdecnordvand1)
            ti18vf = (bf - wf)/(qf - af)
            siconccalc = (T22 - TPdecnordvand1)/(ti18vf - TPdecnordvand1)
            RTMfcomisoDec.append([lat,lon,siconccalc,T31,T22])
        if lat < 0:
            af = (fyDec31s - myDec31s)/(fyDec22s - myDec22s)
            bf = (myDec31s - af*myDec22s)
            qf = (T31 - TPdecsydvand2)/(T22 - TPdecsydvand1)
            wf = (TPdecsydvand2 - qf*TPdecsydvand1)
            ti18vf = (bf - wf)/(qf - af)
            siconccalc = (T22 - TPdecsydvand1)/(ti18vf - TPdecsydvand1)
            RTMfcomisoDec.append([lat,lon,siconccalc,T31,T22])

for x in range(len(RTMJuni)):
    lat = RTMJuni[x,2]
    lon = RTMJuni[x,3]
    T31 = RTMJuni[x,0]
    T22 = RTMJuni[x,1]
    sst = RTMDecember[x,4]
    if T31 > 100 and T22 > 100 and sst < 278:
        if lat > 0:
            af = (fyJun31n - myJun31n)/(fyJun22n - myJun22n)
            bf = (myJun31n - af*myJun22n)
            qf = (T31 - TPjunnordvand2)/(T22 - TPjunnordvand1)
            wf = (TPjunnordvand2 - qf*TPjunnordvand1)
            ti18vf = (bf - wf)/(qf - af)
            siconccalc = (T22 - TPjunnordvand1)/(ti18vf - TPjunnordvand1)
            RTMfcomisoJun.append([lat,lon,siconccalc,T31,T22])
        if lat < 0:
            af = (fyJun31s - myJun31s)/(fyJun22s - myJun22s)
            bf = (myJun31s - af*myJun22s)
            qf = (T31 - TPjunsydvand2)/(T22 - TPjunsydvand1)
            wf = (TPjunsydvand2 - qf*TPjunsydvand1)
            ti18vf = (bf - wf)/(qf - af)
            siconccalc = (T22 - TPjunsydvand1)/(ti18vf - TPjunsydvand1)
            RTMfcomisoJun.append([lat,lon,siconccalc,T31,T22])

RTMfcomisoDec = np.array(RTMfcomisoDec)
RTMfcomisoJun = np.array(RTMfcomisoJun)

with open('RTMfcomisoDec.pkl', 'wb') as f:
    pickle.dump(RTMfcomisoDec, f)
with open('RTMfcomisoJun.pkl', 'wb') as f:
    pickle.dump(RTMfcomisoJun, f)

#%% Usikkerhed singlechannel siconc

# stddecnordvand1
# stddecnordvand2
# stddecsydvand1
# stddecsydvand2
# stdjunnordvand1
# stdjunnordvand2
# stdjunsydvand1
# stdjunsydvand2

#TPdecnordis1 
#TPdecnordis2
#TPdecnordvand1 
#TPdecnordvand2 
#TPdecsydis1 
#TPdecsydis2 
#TPdecsydvand1 
#TPdecsydvand2 

#TPjunnordis1 
#TPjunnordis2 
#TPjunnordvand1 
#TPjunnordvand2 
#TPjunsydis1 
#TPjunsydis2 
#TPjunsydvand1
#TPjunsydvand2

#DecSiconc31n 
#DecSiconc22n 
#DecSiconc31s 
#DecSiconc22s 
#JunSiconc31n 
#JunSiconc22n 
#JunSiconc31s
#JunSiconc22s

decTB31n = []
decTB22n = []
decTB31s = []
decTB22s = []

for x in range(len(RTMDecember)):
    if RTMDecember[x,2] > 0:
        decTB31n.append(RTMDecember[x,0])
        decTB22n.append(RTMDecember[x,1])
    if RTMDecember[x,2] < 0:
        decTB31s.append(RTMDecember[x,0])
        decTB22s.append(RTMDecember[x,1])

decstd31n = np.nanstd(decTB31n)
decstd22n = np.nanstd(decTB22n)
decstd31s = np.nanstd(decTB31s)
decstd22s = np.nanstd(decTB22s)

eps = 0.8

c = []
start_value = 0
end_value = 100
step_size = 0.01

current_value = start_value
while current_value <= end_value:
    c.append(current_value)
    current_value += step_size

DCdecn22 = []
DCdecn31 = []
DCdecs22 = []
DCdecs31 = []


for x in range(len(c)):
    C = c[x]
    diffTB = 1/((TPdecnordis1)-(TPdecnordvand1))
    diffTBw = (-(1-C))/(TPdecnordis1-TPdecnordvand1)
    diffeps = -(C*TPdecnordis1/eps)/(TPdecnordis1-TPdecnordvand1)
    diffTBi = -(C*eps)/(TPdecnordis1-TPdecnordvand1)
    
    dC = ((diffTB*decstd22n)**2+(diffTBw*stddecnordvand1)**2+((diffTBi)*stddecnordis1)**2)**(1/2)
    DCdecn22.append([C,dC])
    
DCdecn22 = np.array(DCdecn22)


for x in range(len(c)):
    C = c[x]
    diffTB = 1/((TPdecnordis2)-(TPdecnordvand2))
    diffTBw = (-(1-C))/(TPdecnordis2-TPdecnordvand2)
    diffeps = -(C*TPdecnordis2)/(TPdecnordis2-TPdecnordvand2)
    diffTBi = -(C*eps)/(TPdecnordis2-TPdecnordvand2)
    
    dC = ((diffTB*decstd31n)**2+(diffTBw*stddecnordvand2)**2+((diffTBi)*stddecnordis2)**2)**(1/2)
    DCdecn31.append([C,dC])
    
DCdecn31 = np.array(DCdecn31)


for x in range(len(c)):
    C = c[x]
    diffTB = 1/((TPdecsydis1)-(TPdecsydvand1))
    diffTBw = (-(1-C))/(TPdecsydis1-TPdecsydvand1)
    diffeps = -(C*TPdecsydis1)/(TPdecsydis1-TPdecsydvand1)
    diffTBi = -(C*eps)/(TPdecsydis1-TPdecsydvand1)
    
    dC = ((diffTB*decstd22s)**2+(diffTBw*stddecsydvand1)**2+((diffTBi)*stddecsydis1)**2)**(1/2)
    DCdecs22.append([C,dC])
    
DCdecs22 = np.array(DCdecs22)


for x in range(len(c)):
    C = c[x]
    diffTB = 1/((TPdecsydis2)-(TPdecsydvand2))
    diffTBw = (-(1-C))/(TPdecsydis2-TPdecsydvand2)
    diffeps = -(C*TPdecsydis2)/(TPdecsydis2-TPdecsydvand2)
    diffTBi = -(C*eps)/(TPdecsydis2-TPdecsydvand2)
    
    dC = ((diffTB*decstd31s)**2+(diffTBw*stddecsydvand2)**2+((diffTBi)*stddecsydis2)**2)**(1/2)
    DCdecs31.append([C,dC])
    
DCdecs31 = np.array(DCdecs31)

sizes=np.ones(len(DCdecn22))*0.1

plt.plot(DCdecn22[:,0],DCdecn22[:,1],color='blue')
plt.plot(DCdecn31[:,0],DCdecn31[:,1],color='green')
plt.plot(DCdecs22[:,0],DCdecs22[:,1],color='blue',linestyle='--')
plt.plot(DCdecs31[:,0],DCdecs31[:,1],color='green',linestyle='--')
plt.title("Usikkerhed for december")
plt.xlabel('SIC')
plt.ylabel('Usikkerhed i iskoncentration')
plt.legend(["Nordlige data 22GHz","Nordlige data 31GHz","Sydlige data 22GHz","Sydlige data 31GHz"])
plt.show(1)



junTB31n = []
junTB22n = []
junTB31s = []
junTB22s = []

for x in range(len(RTMJuni)):
    if RTMJuni[x,2] > 0:
        junTB31n.append(RTMJuni[x,0])
        junTB22n.append(RTMJuni[x,1])
    if RTMJuni[x,2] < 0:
        junTB31s.append(RTMJuni[x,0])
        junTB22s.append(RTMJuni[x,1])

junstd31n = np.nanstd(junTB31n)
junstd22n = np.nanstd(junTB22n)
junstd31s = np.nanstd(junTB31s)
junstd22s = np.nanstd(junTB22s)


eps = 0.8

c = []
start_value = 0
end_value = 100
step_size = 0.01

current_value = start_value
while current_value <= end_value:
    c.append(current_value)
    current_value += step_size

DCjunn22 = []
DCjunn31 = []
DCjuns22 = []
DCjuns31 = []


for x in range(len(c)):
    C = c[x]
    diffTB = 1/((TPjunnordis1)-(TPjunnordvand1))
    diffTBw = (-(1-C))/(TPjunnordis1-TPjunnordvand1)
    diffeps = -(C*TPjunnordis1)/(TPjunnordis1-TPjunnordvand1)
    diffTBi = -(C*eps)/(TPjunnordis1-TPjunnordvand1)
    
    dC = ((diffTB*junstd22n)**2+(diffTBw*stdjunnordvand1)**2+((diffTBi)*stdjunnordis1)**2)**(1/2)
    DCjunn22.append([C,dC])
    
DCjunn22 = np.array(DCjunn22)


for x in range(len(c)):
    C = c[x]
    diffTB = 1/((TPjunnordis2)-(TPjunnordvand2))
    diffTBw = (-(1-C))/(TPjunnordis2-TPjunnordvand2)
    diffeps = -(C*TPjunnordis2)/(TPjunnordis2-TPjunnordvand2)
    diffTBi = -(C*eps)/(TPjunnordis2-TPjunnordvand2)
    
    dC = ((diffTB*junstd31n)**2+(diffTBw*stdjunnordvand2)**2+((diffTBi)*stdjunnordis2)**2)**(1/2)
    DCjunn31.append([C,dC])
    
DCjunn31 = np.array(DCjunn31)


for x in range(len(c)):
    C = c[x]
    diffTB = 1/((TPjunsydis1)-(TPjunsydvand1))
    diffTBw = (-(1-C))/(TPjunsydis1-TPjunsydvand1)
    diffeps = -(C*TPjunsydis1)/(TPjunsydis1-TPjunsydvand1)
    diffTBi = -(C*eps)/(TPjunsydis1-TPjunsydvand1)
    
    dC = ((diffTB*junstd22s)**2+(diffTBw*stdjunsydvand1)**2+((diffTBi)*stdjunsydis1)**2)**(1/2)
    DCjuns22.append([C,dC])
    
DCjuns22 = np.array(DCjuns22)


for x in range(len(c)):
    C = c[x]
    diffTB = 1/((TPjunsydis2)-(TPjunsydvand2))
    diffTBw = (-(1-C))/(TPjunsydis2-TPjunsydvand2)
    diffeps = -(C*TPjunsydis2)/(TPjunsydis2-TPjunsydvand2)
    diffTBi = -(C*eps)/(TPjunsydis2-TPjunsydvand2)
    
    dC = ((diffTB*junstd31s)**2+(diffTBw*stdjunsydvand2)**2+((diffTBi)*stdjunsydis2)**2)**(1/2)
    DCjuns31.append([C,dC])
    
DCjuns31 = np.array(DCjuns31)

sizes=np.ones(len(DCjunn22))*0.1

plt.plot(DCjunn22[:,0],DCjunn22[:,1],color='blue')
plt.plot(DCjunn31[:,0],DCjunn31[:,1],color='green')
plt.plot(DCjuns22[:,0],DCjuns22[:,1],color='blue',linestyle='--')
plt.plot(DCjuns31[:,0],DCjuns31[:,1],color='green',linestyle='--')
plt.title("Usikkerhed for juni")
plt.xlabel('SIC')
plt.ylabel('Usikkerhed i iskoncentration')
plt.legend(["Nordlige data 22GHz","Nordlige data 31GHz","Sydlige data 22GHz","Sydlige data 31GHz"])
plt.show(2)

#%% Standardafvigelser på tiepoints med 2kanals usikkerhed

#lat,lon,siconccalc,T31,T22

with open('RTMfcomisoDec.pkl', 'rb') as f:
    RTMDecember = pickle.load(f)

with open('RTMfcomisoJun.pkl', 'rb') as f:
    RTMJuni = pickle.load(f)

Decemberisnord = []
Decembervandnord = []
Decemberissyd = []
Decembervandsyd = []

for x in range(len(RTMDecember)):
    sic = RTMDecember[x,2]
    lat = RTMDecember[x,0]
    if sic > 1:
        sic = 1
    if sic < 0:
        sic = 0
    if lat>0:
        if sic > 0.9:
            Decemberisnord.append(sic)
        if sic < 0.3:
            Decembervandnord.append(sic)
    if lat<0:
        if sic > 0.9:
            Decemberissyd.append(sic)
        if sic < 0.3:
            Decembervandsyd.append(sic)

stddecisnord = np.std(Decemberisnord)
stddecvandnord = np.std(Decembervandnord)
stddecissyd = np.std(Decemberissyd)
stddecvandsyd = np.std(Decembervandsyd)

Juniisnord = []
Junivandnord = []
Juniissyd = []
Junivandsyd = []

for x in range(len(RTMJuni)):
    sic = RTMJuni[x,2]
    lat = RTMJuni[x,0]
    if sic > 1:
        sic = 1
    if sic < 0:
        sic = 0
    if lat>0:
        if sic > 0.9:
            Juniisnord.append(sic)
        if sic < 0.3:
            Junivandnord.append(sic)
    if lat<0:
        if sic > 0.9:
            Juniissyd.append(sic)
        if sic < 0.3:
            Junivandsyd.append(sic)

stdjunisnord = np.std(Juniisnord)
stdjunvandnord = np.std(Junivandnord)
stdjunissyd = np.std(Juniissyd)
stdjunvandsyd = np.std(Junivandsyd)


#%%
c = []
start_value = 0
end_value = 100
step_size = 0.01

current_value = start_value
while current_value <= end_value:
    c.append(current_value)
    current_value += step_size

eice = stddecisnord
ew = stddecvandnord

Usikkerheddecnord = []

for x in range(len(c)):
    C = c[x]
    usik = np.sqrt((1-C)**2*ew**2+C**2*eice**2)
    Usikkerheddecnord.append([C,usik])
    
Usikkerheddecnord = np.array(Usikkerheddecnord)

eice = stddecissyd
ew = stddecvandsyd

Usikkerheddecsyd = []

for x in range(len(c)):
    C = c[x]
    usik = np.sqrt((1-C)**2*ew**2+C**2*eice**2)
    Usikkerheddecsyd.append([C,usik])
    
Usikkerheddecsyd = np.array(Usikkerheddecsyd)
  
eice = stdjunisnord
ew = stdjunvandnord

Usikkerhedjunnord = []

for x in range(len(c)):
    C = c[x]
    usik = np.sqrt((1-C)**2*ew**2+C**2*eice**2)
    Usikkerhedjunnord.append([C,usik])
    
Usikkerhedjunnord = np.array(Usikkerhedjunnord)

eice = stdjunissyd
ew = stdjunvandsyd

Usikkerhedjunsyd = []

for x in range(len(c)):
    C = c[x]
    usik = np.sqrt((1-C)**2*ew**2+C**2*eice**2)
    Usikkerhedjunsyd.append([C,usik])
    
Usikkerhedjunsyd = np.array(Usikkerhedjunsyd)


plt.plot(Usikkerheddecnord[:,0],Usikkerheddecnord[:,1],color='green')
plt.plot(Usikkerheddecsyd[:,0],Usikkerheddecsyd[:,1],color='green',linestyle='--')
plt.plot(Usikkerhedjunnord[:,0],Usikkerhedjunnord[:,1],color='blue')
plt.plot(Usikkerhedjunsyd[:,0],Usikkerhedjunsyd[:,1],color='blue',linestyle='--')
plt.title("Usikkerhed for 2-kanals iskoncentration")
plt.xlabel('Iskoncentration')
plt.ylabel('Usikkerhed i iskoncentration')
plt.legend(["December Nord","December Syd","Juni Nord","Juni Syd"])
plt.show()