# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:22:07 2023

@author: janus
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:20:09 2023

@author: janus
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:21:15 2023

@author: janus
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import glob
import pickle


with open('nordsiconcchn1hverset.pkl', 'rb') as f:
    Siconc1nhverset = pickle.load(f)

with open('sydsiconcchn1hverset.pkl', 'rb') as f:
    Siconc1shverset = pickle.load(f)

with open('nordsiconcchn2hverset.pkl', 'rb') as f:
    Siconc2nhverset = pickle.load(f)

with open('sydsiconcchn2hverset.pkl', 'rb') as f:
    Siconc2shverset = pickle.load(f)    

for sublist in Siconc1nhverset:
    # Create a new list to store the filtered rows
    filtered_rows = []
    
    # Iterate over each row in the sublist
    for row in sublist:
        # Check if the value in the third column is 0.9 or above
        if row[2] >= 0.9:
            filtered_rows.append(row)
    
    # Replace the original sublist with the filtered rows
    sublist[:] = filtered_rows

for sublist in Siconc1shverset:
    # Create a new list to store the filtered rows
    filtered_rows = []
    
    # Iterate over each row in the sublist
    for row in sublist:
        # Check if the value in the third column is 0.9 or above
        if row[2] >= 0.9:
            filtered_rows.append(row)
    
    # Replace the original sublist with the filtered rows
    sublist[:] = filtered_rows

for sublist in Siconc2nhverset:
    # Create a new list to store the filtered rows
    filtered_rows = []
    
    # Iterate over each row in the sublist
    for row in sublist:
        # Check if the value in the third column is 0.9 or above
        if row[2] >= 0.9:
            filtered_rows.append(row)
    
    # Replace the original sublist with the filtered rows
    sublist[:] = filtered_rows

for sublist in Siconc2shverset:
    # Create a new list to store the filtered rows
    filtered_rows = []
    
    # Iterate over each row in the sublist
    for row in sublist:
        # Check if the value in the third column is 0.9 or above
        if row[2] >= 0.9:
            filtered_rows.append(row)
    
    # Replace the original sublist with the filtered rows
    sublist[:] = filtered_rows

Sorteretsiconc1n = []
Sorteretsiconc1s = []
Sorteretsiconc2n = []
Sorteretsiconc2s = []

for x in range(6):
    sic1n = sorted(Siconc1nhverset[x],key=lambda y: y[3])
    sic1s = sorted(Siconc1shverset[x],key=lambda y: y[3])
    sic2n = sorted(Siconc2nhverset[x],key=lambda y: y[3])
    sic2s = sorted(Siconc2shverset[x],key=lambda y: y[3])
    Sorteretsiconc1n.append(sic1n)
    Sorteretsiconc1s.append(sic1s)
    Sorteretsiconc2n.append(sic2n)
    Sorteretsiconc2s.append(sic2s)



#Udregn vaerdi for first year og multiyear ice
tp1nfirstyear = []
tp1sfirstyear = []
tp1nmultiyear = []
tp1smultiyear = []
tp2nfirstyear = []
tp2sfirstyear = []
tp2nmultiyear = []
tp2smultiyear = []

for x in range(6):
    n1n = int(len(Sorteretsiconc1n[x]) * 0.2)
    n2s = int(len(Sorteretsiconc2s[x]) * 0.2)
    n1s = int(len(Sorteretsiconc1s[x]) * 0.2)
    n2n = int(len(Sorteretsiconc2n[x]) * 0.2)
    
    sumsic1nmy = sum(sublist[3] for sublist in Sorteretsiconc1n[x][:n1n])
    average1nmy = sumsic1nmy / n1n
    tp1nmultiyear.append(average1nmy)
    
    sumsic1smy = sum(sublist[3] for sublist in Sorteretsiconc1s[x][:n1s])
    average1smy = sumsic1smy / n1s
    tp1smultiyear.append(average1smy)
    
    sumsic2nmy = sum(sublist[3] for sublist in Sorteretsiconc2n[x][:n2n])
    average2nmy = sumsic2nmy / n2n
    tp2nmultiyear.append(average2nmy)
    
    sumsic2smy = sum(sublist[3] for sublist in Sorteretsiconc2s[x][:n2s])
    average2smy = sumsic2smy / n2s
    tp2smultiyear.append(average2smy)
    
    
    sumsic1nfy = sum(sublist[3] for sublist in Sorteretsiconc1n[x][-n1n:])
    average1nfy = sumsic1nfy / n1n
    tp1nfirstyear.append(average1nfy)
    
    sumsic1sfy = sum(sublist[3] for sublist in Sorteretsiconc1s[x][-n1s:])
    average1sfy = sumsic1sfy / n1s
    tp1sfirstyear.append(average1sfy)
    
    sumsic2nfy = sum(sublist[3] for sublist in Sorteretsiconc2n[x][-n2n:])
    average2nfy = sumsic2nfy / n2n
    tp2nfirstyear.append(average2nfy)
    
    sumsic2sfy = sum(sublist[3] for sublist in Sorteretsiconc2s[x][-n2s:])
    average2sfy = sumsic2sfy / n2s
    tp2sfirstyear.append(average2sfy)

with open('tp1nfirstyear.pkl', 'wb') as f:
    pickle.dump(tp1nfirstyear, f)
with open('tp1sfirstyear.pkl', 'wb') as f:
    pickle.dump(tp1sfirstyear, f)
with open('tp2nfirstyear.pkl', 'wb') as f:
    pickle.dump(tp2nfirstyear, f)
with open('tp2sfirstyear.pkl', 'wb') as f:
    pickle.dump(tp2sfirstyear, f)
with open('tp1nmultiyear.pkl', 'wb') as f:
    pickle.dump(tp1nmultiyear, f)
with open('tp1smultiyear.pkl', 'wb') as f:
    pickle.dump(tp1smultiyear, f)
with open('tp2nmultiyear.pkl', 'wb') as f:
    pickle.dump(tp2nmultiyear, f)
with open('tp2smultiyear.pkl', 'wb') as f:
    pickle.dump(tp2smultiyear, f)


with open('tpnv1.pkl', 'rb') as f:
    tpnv1 = pickle.load(f)

with open('tpnv2.pkl', 'rb') as f:
    tpnv2 = pickle.load(f)

with open('tpsv1.pkl', 'rb') as f:
    tpsv1 = pickle.load(f)

with open('tpsv2.pkl', 'rb') as f:
    tpsv2 = pickle.load(f) 

dec72nordfyc2 = [tp2nfirstyear[0],tp2nfirstyear[1],tp2nfirstyear[2]]
dec72nordmyc2 = [tp2nmultiyear[0],tp2nmultiyear[1],tp2nmultiyear[2]]
dec72nordfyc1 = [tp1nfirstyear[0],tp1nfirstyear[1],tp1nfirstyear[2]]
dec72nordmyc1 = [tp1nmultiyear[0],tp1nmultiyear[1],tp1nmultiyear[2]]
dectpnv2 = [tpnv2[0],tpnv2[1],tpnv2[2]]
dectpnv1 = [tpnv1[0],tpnv1[1],tpnv1[2]]


jun73nordfyc2 = [tp2nfirstyear[3],tp2nfirstyear[4],tp2nfirstyear[5]]
jun73nordmyc2 = [tp2nmultiyear[3],tp2nmultiyear[4],tp2nmultiyear[5]]
jun73nordfyc1 = [tp1nfirstyear[3],tp1nfirstyear[4],tp1nfirstyear[5]]
jun73nordmyc1 = [tp1nmultiyear[3],tp1nmultiyear[4],tp1nmultiyear[5]]
juntpnv2 = [tpnv2[3],tpnv2[4],tpnv2[5]]
juntpnv1 = [tpnv1[3],tpnv1[4],tpnv1[5]]
 
dec72sydfyc2 = [tp2sfirstyear[0],tp2sfirstyear[1],tp2sfirstyear[2]]
dec72sydmyc2 = [tp2smultiyear[0],tp2smultiyear[1],tp2smultiyear[2]]
dec72sydfyc1 = [tp1sfirstyear[0],tp1sfirstyear[1],tp1sfirstyear[2]]
dec72sydmyc1 = [tp1smultiyear[0],tp1smultiyear[1],tp1smultiyear[2]]
dectpsv2 = [tpsv2[0],tpsv2[1],tpsv2[2]]
dectpsv1 = [tpsv1[0],tpsv1[1],tpsv1[2]]


jun73sydfyc2 = [tp2sfirstyear[3],tp2sfirstyear[4],tp2sfirstyear[5]]
jun73sydmyc2 = [tp2smultiyear[3],tp2smultiyear[4],tp2smultiyear[5]]
jun73sydfyc1 = [tp1sfirstyear[3],tp1sfirstyear[4],tp1sfirstyear[5]]
jun73sydmyc1 = [tp1smultiyear[3],tp1smultiyear[4],tp1smultiyear[5]]
juntpsv2 = [tpsv2[3],tpsv2[4],tpsv2[5]]
juntpsv1 = [tpsv1[3],tpsv1[4],tpsv1[5]]


plt.scatter(dec72nordfyc2,dec72nordfyc1,color='red')
plt.scatter(dec72nordmyc2,dec72nordmyc1,color='black')
plt.scatter(dectpnv2,dectpnv1,color='blue')
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')
plt.title("December 72 Tiepoints Nord")
plt.show(1)

plt.scatter(jun73nordfyc2,jun73nordfyc1,color='red')
plt.scatter(jun73nordmyc2,jun73nordmyc1,color='black')
plt.scatter(juntpnv2,juntpnv1,color='blue')
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')
plt.title("Juni 73 Tiepoints Nord")
plt.show(2)


plt.scatter(dec72sydfyc2,dec72sydfyc1,color='red')
plt.scatter(dec72sydmyc2,dec72sydmyc1,color='black')
plt.scatter(dectpsv2,dectpsv1,color='blue')
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')
plt.title("December 72 Tiepoints Syd")
plt.show(3)

plt.scatter(jun73sydfyc2,jun73sydfyc1,color='red')
plt.scatter(jun73sydmyc2,jun73sydmyc1,color='black')
plt.scatter(juntpsv2,juntpsv1,color='blue')
plt.xlabel('TB 31.4 GHz [K]')
plt.ylabel('TB 22.235 GHz [K]')
plt.title("Juni 73 Tiepoints Syd")
plt.show(4)

