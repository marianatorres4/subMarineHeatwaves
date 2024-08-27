# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:24:47 2024

@author: 24048369
"""

import xarray as xr
import pandas as pd
import numpy as np 
# import ocetrac
from matplotlib import pyplot as plt
import datetime
import netCDF4 
from datetime import date
from datetime import timedelta
import glob
from matplotlib.colors import ListedColormap
from datetime import date, timedelta, datetime

sys.path.append('C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL')

from ocetrac3D import Tracker3D
# import ocetrac3D


folder = 'C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/'
file_TAS   = "GLORYS_data_TAS_(1).nc"
file_NAA   = "GLORYS_data_NA.nc"

ds_TAS = xr.open_dataset(folder + file_TAS)
ds_NAA = xr.open_dataset(folder + file_NAA)


# ds_NAA
# ds_TAS


#%%

tos = ds_TAS.sel(time = slice ('2015-01-01','2017-01-01'))
climatology = tos.groupby(tos.time.dt.month).mean()
anomaly = tos.groupby(tos.time.dt.month) - climatology
# Rechunk time dim
if tos.chunks:
    tos = tos.chunk({'time': -1})
percentile = .9
threshold = tos.groupby(tos.time.dt.month).quantile(percentile, dim='time', keep_attrs=True, skipna=True)
hot_water = anomaly.groupby(tos.time.dt.month).where(tos.groupby(tos.time.dt.month)>threshold)
threshold

#%%
mask_ocean = 1 * np.ones(tos.thetao.shape[1:]) * np.isfinite(tos.isel(time=0))
mask_land = 0 * np.ones(tos.thetao.shape[1:]) * np.isnan(tos.isel(time=0))
mask = mask_ocean + mask_land
mask.thetao.sel(depth = 0,method = 'nearest').plot()


time_array = tos.time.data
lon,lat = tos.longitude.data, tos.latitude.data

mask1 = mask.thetao.sel(depth = 0,method = 'nearest')



plt.figure(figsize=(16,3),dpi = 200)
t = 200
ax1 = plt.subplot(121)
anomaly.thetao.sel(time= time_array[t],depth = 50, method = 'nearest').plot(cmap='RdBu_r', vmin=-2, vmax=2, extend='both') 
ax1.set_aspect('equal')
ax.contourf(lon,lat, mask1.where(mask1 ==0)   , colors='k') 

ax2 = plt.subplot(122);
hot_water.thetao.sel(time= time_array[t],depth = 50, method = 'nearest').plot(cmap='Reds', vmin=0);
ax.contourf(lon,lat, mask1.where(mask1 ==0)   , colors='k') 
ax2.set_aspect('equal')




    
#%%

#### try 3D ocetrack

hot_water_short = hot_water.sel(time = slice('2015-11-01','2016-02-01')).sel( depth = slice(0,40))#.mean('latitude')
mask2 = mask.sel( depth = slice(0,40))

Tracker = Tracker3D(hot_water_short.thetao, mask2.thetao, radius=2, min_size_quartile=0.75,
                          timedim = 'time', xdim = 'longitude', ydim='latitude', ddim = 'depth',
                          positive=True)
blobs = Tracker.track()

#%%
blobs.attrs





maxl = int(np.nanmax(blobs.values))
Nr_blobs = blobs.attrs['final objects tracked']
base_cmap = plt.get_cmap('gist_rainbow')
colors = base_cmap(np.linspace(0, 1, Nr_blobs))
custom_cmap = ListedColormap(colors)

cm = ListedColormap(np.random.random(size=(maxl, 3)).tolist())
lon = hot_water_short.longitude.data
lat = hot_water_short.latitude.data
t = 8500,
for t in range(65,75): #len(hot_water_short.time.data)):
    
    fig, axs = plt.subplots(ncols=2, nrows =8, figsize=(6,10),dpi = 150)
    for id_d,d in enumerate([0,2,4,6,8,10,12,14]):
        
        
        ax = axs[id_d,0]
        if id_d ==0:
            ax.set_title(str(hot_water_short.time[t].data)[:10])
        
        ax.set_ylabel('depth:' + str(np.round(hot_water_short.isel(depth = d).depth.data)))
        
        levels = np.linspace(0,Nr_blobs,Nr_blobs + 1)
        ax.contourf(lon,lat,mask2.thetao.isel(depth = d).where(mask2.thetao.isel(depth = d)==0),colors='k') 
        
        cm = ax.contourf(lon,lat,blobs.isel(time=t, depth=d).data, cmap = custom_cmap, #vmin=1, vmax=Nr_blobs
                        levels = levels)
        
        cbaxes = fig.add_axes([0.1, -0.05, 0.8, 0.019])
        
    
        cb = fig.colorbar(cm, orientation='horizontal',cax=cbaxes,aspect = 0.5,shrink=0.8, ax = ax1, 
                          ticks = levels,)
        cb.set_label('Label', size=12)
        cb.ax.tick_params(labelsize=6)
        
        
        
        ax.set_aspect('equal')
        ax = axs[id_d,1]
        
        ax.contourf(lon,lat,hot_water_short.thetao.isel(time=t,depth = d).data, cmap = 'Reds')
        ax.contourf(lon,lat,mask2.thetao.isel(depth = d).where(mask2.thetao.isel(depth = d)==0),colors='k') 
        ax.set_aspect('equal');
    
    

    plt.savefig(folder + 'depth_' +str(hot_water_short.time[t].data)[:10] + '.png', bbox_inches='tight',dpi = 300)
    plt.savefig(folder + 'depth_' +str(hot_water_short.time[t].data)[:10] + '.pdf', bbox_inches='tight',dpi = 300)




























