# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:43:01 2023

@author: noahe
"""
import numpy as np
import pickle


#with open('C:/Users/noahe/OneDrive/Dokumenter/DTU/Fagprojekt/RTMfcomisoDec.pkl','rb') as f:
with open('C:/Users/noahe/OneDrive/Dokumenter/DTU/Fagprojekt/RTMfcomisoJun.pkl','rb') as f:
    ds = pickle.load(f)

data = np.array(ds)

T31 = data[:,3]
T22 = data[:,4]
lat =data[:,0]
lon = data[:,1]

nyGR = []


for x in range (len(data)):
    GR = (T31[x]-T22[x])/(T31[x]+T22[x])
    nyGR.append([lat[x],lon[x],GR])

with open('GRJun.pkl', 'wb') as f:
    pickle.dump(nyGR, f)

