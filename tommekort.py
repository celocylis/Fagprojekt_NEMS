import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
import numpy as np
import scipy
from scipy.interpolate import griddata
import pickle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#North pole:
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','ocean','50m',facecolor=[0.8,0.8,0.8]))
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, 90, 60], ccrs.PlateCarree())

ax.coastlines()

plt.show()


#North pole:
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.add_feature(cfeature.NaturalEarthFeature('physical','ocean','50m',facecolor=[0.8,0.8,0.8]))
ax.add_feature(cfeature.NaturalEarthFeature('physical','land','50m',facecolor='grey'))
ax.set_extent([0, 360, -60, -90], ccrs.PlateCarree())

ax.coastlines()

plt.show()