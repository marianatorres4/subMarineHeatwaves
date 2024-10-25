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
import sys
import os 
import moviepy.video.io.ImageSequenceClip
import cmocean
import imageio.v2 as imageio
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
from matplotlib.colors import LinearSegmentedColormap
import string

sys.path.append('C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/Ocetrac')

from ocetrac_3D_and_4D import Tracker3D, Tracker4D
# import ocetrac3D


folder = 'C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/Ocetrac/'
file_WA  = "GLORYS_data_WA_ocetrac_very_big.nc"
file = 'C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/GLORYS_data_WA_spatial_extent_analysis.nc'
ds_WA = xr.open_dataset(file)

# ds_WA = xr.open_dataset(folder + file_WA)

#%%

# # Process in smaller time batches to reduce memory usage
# batch_size = 500  # Change this to an appropriate batch size

# lat_factor = int(0.15 / (ds_WA.latitude[1] - ds_WA.latitude[0]).values)
# lon_factor = int(0.15 / (ds_WA.longitude[1] - ds_WA.longitude[0]).values)



# coarsened_list = []


# new_depth = np.array([0.5,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200])


# # Loop over time slices in batches
# for i in range(0,len(ds_WA.time) , batch_size):  # 600
#     ds_chunk = ds_WA.isel(time=slice(i, i + batch_size))
    
#     ds_coarsened_chunk = ds_chunk.coarsen(
#         latitude=lat_factor, 
#         longitude=lon_factor, 
#         boundary='trim' ).mean()
    
#     ds_coarsened_chunk_new_depth = ds_coarsened_chunk.interp(depth = new_depth)
    
#     coarsened_list.append(ds_coarsened_chunk_new_depth)
    
    
# ds_coarsened = xr.concat(coarsened_list, dim='time')



# #%%

# tos = ds_coarsened
# del ds_coarsened, ds_WA, ds_chunk, ds_coarsened_chunk_new_depth, ds_coarsened_chunk

 #%%
tos = ds_WA 
ds_WA_small = ds_WA.sel(time = slice(pd.to_datetime('2010-05-01'),pd.to_datetime('2011-12-01')))


print(datetime.now())

climatology = tos.groupby(tos.time.dt.dayofyear).mean()

anomaly = ds_WA_small.groupby(ds_WA_small.time.dt.dayofyear) - climatology

#%%

percentile = .9
threshold = tos.groupby(tos.time.dt.dayofyear).quantile(percentile, dim='time', keep_attrs=True, skipna=True)
threshold_small = threshold.sel(time = slice(pd.datetime('2010-05-01'),pd.datetime('2011-12-01')))

hot_water = anomaly.groupby(ds_WA_small.time.dt.dayofyear).where(ds_WA_small.groupby(ds_WA_small.time.dt.dayofyear)>threshold_small)
threshold
print(datetime.now())



#%%
mask_ocean = 1 * np.ones(tos.thetao.shape[1:]) * np.isfinite(tos.isel(time=0))
mask_land = 0 * np.ones(tos.thetao.shape[1:]) * np.isnan(tos.isel(time=0))
mask = mask_ocean + mask_land
mask.thetao.sel(depth = 0,method = 'nearest').plot()

#%%
time_array = tos.time.data
lon,lat = tos.longitude.data, tos.latitude.data

d = 100
mask1 = mask.thetao.sel(depth = d,method = 'nearest')

mask_surface = mask.thetao.sel(depth = 0,method = 'nearest')


plt.figure(figsize=(12,5),dpi = 200)
t = 6600
ax1 = plt.subplot(121)
anomaly.thetao.sel(time= time_array[t],depth = d, method = 'nearest').plot(cmap='RdBu_r', vmin=-2, vmax=2, extend='both') 
ax1.set_aspect('equal')
ax1.contourf(lon,lat, mask1.where(mask1 ==0)   , colors='grey') 
ax1.contourf(lon,lat, mask_surface.where(mask_surface ==0)   , colors='k') 

ax2 = plt.subplot(122);
hot_water.thetao.sel(time= time_array[t],depth = d, method = 'nearest').plot(cmap='Reds', vmin=0,vmax=2, extend='max')
ax2.contourf(lon,lat, mask1.where(mask1 ==0)   , colors='grey') 
ax2.contourf(lon,lat, mask_surface.where(mask_surface ==0)   , colors='k') 
ax2.set_aspect('equal')




    
# #%%

# #### try 3D ocetrack

# # hot_water_short = hot_water.sel(time = slice('2010-11-01','2011-04-01')).sel( depth = [0,10,20,30,50,100], method = 'nearest')#.mean('latitude')
# # mask2 = mask.sel( depth = [0,10,20,30,50,100], method = 'nearest')

# d_sel = 125
# hot_water_short = hot_water.sel(time = slice('2010-10-01','2011-05-01')).sel( depth = d_sel, method = 'nearest')
# mask2 = mask.sel( depth = d_sel, method = 'nearest')
# mask_surface = mask.sel( depth = 0, method = 'nearest')

# Tracker = Tracker3D(hot_water_short.thetao, mask2.thetao, radius=2, min_size_quartile=0.75,
#                           timedim = 'time', xdim = 'longitude', ydim='latitude',
#                           positive=True)
# blobs = Tracker.track()
# stats = Tracker.collect_surface_stats(blobs)

#%%


# =============================================================================
# 4D Ocetrack 
# =============================================================================
Tracker = Tracker4D(hot_water.sel(time = slice('2010-09-01','2011-06-01')).thetao, mask.thetao, 
                    radius=2, min_size_quartile=0.6,
                          timedim = 'time', xdim = 'longitude', ydim='latitude',ddim = 'depth',
                          positive=True)
blobs4d = Tracker.track()

stats = Tracker.collect_volume_stats(blobs4d)



blobs4d
blobs_dataset4d = xr.Dataset({'Blobs': blobs4d})

folder_saving = 'C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/Ocetrac/'
blobs_dataset4d.to_netcdf(folder_saving + 'merged_blobs_2010-2011_4D.nc')


mask
mask.to_netcdf(folder_saving + 'mask_2010-2011_4D.nc')





#%%


stats.volume_change

idsubset= 7
stat_vol = np.array(stats[stats.id == idsubset].volume_change.iloc[0])
stat_surfarea = np.array(stats[stats.id == idsubset].surface_area_change.iloc[0])
stat_depthint = np.array(stats[stats.id == idsubset].intensity_max_depthlevel.iloc[0])
stat_time = pd.to_datetime(np.array(stats[stats.id == idsubset].dates.iloc[0]))
stat_intmax = np.array(stats[stats.id == idsubset].intensity_max.iloc[0])
stat_intmean = np.array(stats[stats.id == idsubset].intensity_mean.iloc[0])

centroids = np.array(stats[stats.id == idsubset].centroid.iloc[0])
lons_centr, lats_centr = zip(*centroids)


fig, axes = plt.subplots(ncols = 1, nrows = 5,dpi = 300, sharex = True, 
                         figsize = (8,10))
ax = axes[0]
ax.plot(stat_time, -stat_depthint, label = 'Depth of maximum intensity')
mean_d = np.nanmean(-stat_depthint)
ax.axhline(mean_d, color = 'red', lw = 2, label = 'mean depth at ' + str(mean_d))


ax.legend()
ax = axes[1]
ax.plot(stat_time, stat_vol, label = 'Volume')
id_max = np.where(stat_vol == np.max(stat_vol))[0][0]
ax.axvline(stat_time[id_max], color = 'red', lw = 2, label = 'max at ' + str(stat_time[id_max])[:10])

ax.legend()
ax = axes[2]
ax.plot(stat_time, stat_intmax, label = 'Max Intensity')
id_max = np.where(stat_intmax == np.max(stat_intmax))[0]
ax.axvline(stat_time[id_max], color = 'red', lw = 2, label = 'max at' + str(stat_time[id_max][0])[:10])

ax.legend()
ax = axes[3]
ax.plot(stat_time, stat_intmean, label = 'Mean Intensity')
id_max = np.where(stat_intmean == np.max(stat_intmean))[0]
ax.axvline(stat_time[id_max], color = 'red', lw = 2, label = 'max at' + str(stat_time[id_max][0])[:10])
ax.legend()

ax = axes[4]
ax.plot(stat_time, stat_surfarea, label = 'Surface area')
id_max = np.where(stat_surfarea == np.max(stat_surfarea))[0]
ax.axvline(stat_time[id_max], color = 'red', lw = 2, label = 'max at' + str(stat_time[id_max][0])[:10])
ax.legend()

ax.tick_params(axis='x', rotation=90, labelsize=11, labelcolor = 'black') 


fig, ax = plt.subplots(figsize=(10, 10),dpi = 300, subplot_kw={'projection': ccrs.PlateCarree()})

# Add a map feature
ax.coastlines()
# ax.add_feature(cartopy.feature.BORDERS, linestyle=':')

# Plot the points
ax.scatter(lons_centr, lats_centr, color='red', marker='o', transform=ccrs.PlateCarree())

# Set title and labels
ax.set_title('Longitude and Latitude Plot')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_ylim([-35,-25])
ax.set_xlim([105,117.5])
plt.show()




# d_sel = 50
# hot_water_short50 = hot_water.sel(time = slice('2010-11-01','2011-05-01')).sel( depth = d_sel, method = 'nearest')
# mask50 = mask.sel( depth = d_sel, method = 'nearest')
# mask_surface = mask.sel( depth = 0, method = 'nearest')

# Tracker = Tracker3D(hot_water_short50.thetao, mask50.thetao, radius=1, min_size_quartile=0.6,
#                           timedim = 'time', xdim = 'longitude', ydim='latitude',
#                           positive=True)
# blobs50 = Tracker.track()


# d_sel = 100
# hot_water_short100 = hot_water.sel(time = slice('2010-11-01','2011-05-01')).sel( depth = d_sel, method = 'nearest')
# mask100 = mask.sel( depth = d_sel, method = 'nearest')
# mask_surface = mask.sel( depth = 0, method = 'nearest')

# Tracker = Tracker3D(hot_water_short100.thetao, mask100.thetao, radius=2, min_size_quartile=0.6,
#                           timedim = 'time', xdim = 'longitude', ydim='latitude',
#                           positive=True)
# blobs100 = Tracker.track()


#%%

# =============================================================================
# 3D ocetrac
# =============================================================================

file = 'C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/GLORYS_data_WA_ocetrac_whole_surface.nc'
file_clim = 'C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/GLORYS_data_WA_ocetrac_whole_surface_cliamotloy2.nc'

print(datetime.now())
tos = xr.open_dataset(file)
tos = tos.isel(depth = 0 )

tos_clim = xr.open_dataset(file_clim)
tos_clim = tos_clim.isel(depth = 0 )


lon_factor = tos_clim.sizes['longitude'] // 120
lat_factor = tos_clim.sizes['latitude'] // 70

# Apply coarsening
coarsened_clim = tos_clim.coarsen(longitude=lon_factor, latitude=lat_factor, boundary="trim").mean()

lon_factor = tos.sizes['longitude'] // 120
lat_factor = tos.sizes['latitude'] // 70

# Apply coarsening
coarsened_data = tos.coarsen(longitude=lon_factor, latitude=lat_factor, boundary="trim").mean()



print(datetime.now())


#%%
climatology = coarsened_data.groupby(coarsened_data.time.dt.dayofyear).mean()
anomaly = coarsened_data.groupby(coarsened_data.time.dt.dayofyear) - climatology
# # Rechunk time dim
# if tos.chunks:
#     tos = tos.chunk({'time': -1})
percentile = .9
threshold = coarsened_data.groupby(coarsened_data.time.dt.dayofyear).quantile(percentile, dim='time', keep_attrs=True, skipna=True)
hot_water_surface = anomaly.groupby(coarsened_data.time.dt.dayofyear).where(coarsened_data.groupby(coarsened_data.time.dt.dayofyear)>threshold)
threshold
print(datetime.now())

#%%

mask_ocean = 1 * np.ones(coarsened_data.thetao.shape[1:]) * np.isfinite(coarsened_data.isel(time=0))
mask_land = 0 * np.ones(coarsened_data.thetao.shape[1:]) * np.isnan(coarsened_data.isel(time=0))
mask_surface = mask_ocean + mask_land
mask_surface.thetao.plot()
mask_surface = mask_surface.thetao

time_array = coarsened_data.time.data
lon,lat = coarsened_data.longitude.data, coarsened_data.latitude.data





### plot anomaly and hot water 
plt.figure(figsize=(12,5),dpi = 200)
t = 6500
ax1 = plt.subplot(121)
anomaly.thetao.sel(time= time_array[t], method = 'nearest').plot(cmap='RdBu_r', vmin=-2, vmax=2, extend='both') 
ax1.set_aspect('equal')
ax1.contourf(lon,lat, mask_surface.where(mask_surface ==0)   , colors='grey') 
ax2 = plt.subplot(122);
hot_water_surface.thetao.sel(time= time_array[t], method = 'nearest').plot(cmap='Reds', vmin=0,vmax=2, extend='max')
ax2.contourf(lon,lat, mask_surface.where(mask_surface ==0)   , colors='grey') 
ax2.set_aspect('equal')



### call ocetrac
hot_water_short_surface = hot_water_surface.sel(time = slice('2010-10-01','2011-10-01'))


Tracker = Tracker3D(hot_water_short_surface.thetao, mask_surface, radius=3, min_size_quartile=0.8,
                          timedim = 'time', xdim = 'longitude', ydim='latitude',
                          positive=True)
blobs_surface = Tracker.track()
stats = Tracker.collect_surface_stats(blobs_surface)


#%%
# =============================================================================
# crate cummulative intensity heat map.
# =============================================================================


b_blobs = blobs_surface
maskb = b_blobs == 1
time_steps_with_value_3 = maskb.any(dim=['longitude', 'latitude'])


indices_with_value_3 = time_steps_with_value_3.where(time_steps_with_value_3, drop=True).time
time_indices = indices_with_value_3.values

blobs_with_value_3 = b_blobs.sel(time=time_indices)
hotwater_with_value_3 = hot_water_short_surface.sel(time=time_indices)
cum_int = hotwater_with_value_3.sum('time')



### colormap for cum intensity 
cmap = plt.get_cmap('YlOrRd')

n_white = 10  # Adjust this to set how many white colors you want

colors = [(1, 1, 1)] * n_white + [cmap(i) for i in range(cmap.N)]

new_cmap = LinearSegmentedColormap.from_list('CustomYlOrRd', colors, N=cmap.N + n_white)




proj = ccrs.PlateCarree()

fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6,9), dpi = 200,
                      subplot_kw={'projection': proj},gridspec_kw = {'wspace':0.3 })


map_ocean = cfeature.OCEAN.with_scale("10m")
map_land = cfeature.LAND.with_scale("10m")
# ax.coastlines()

ax.set_title('Depth: 0m\n' + str(time_indices[0])[:10] +  ' - ' + str(time_indices[-1])[:10]  , fontsize = 14)
  
fig.subplots_adjust(wspace=0.7)
    


ax.contourf(lon,lat,mask_surface.where(mask_surface==0),colors='grey',zorder = 3) 
ax.contourf(lon,lat,mask_surface.where(mask_surface==0),colors='k',zorder = 3) 

cm = ax.contourf(lon,lat,cum_int.thetao.data,50, cmap = new_cmap )
cbaxes = fig.add_axes([0.95, 0.2, 0.035, 0.6])
cb = fig.colorbar(cm, orientation='vertical',cax=cbaxes,aspect = 0.5,shrink=0.8, ax = ax, 
                   
                  )
cb.set_label('Cumulative Intensity [°C]', size=10)
cb.ax.tick_params(labelsize=9)

lat_watr = -31.6
lon_watr  = 115.0733 
ax.scatter(lon_watr,lat_watr, edgecolor = 'white',facecolor='navy', s = 50)    
ax.set_aspect('equal')



gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.8, color='gray', alpha=0.7, linestyle='--', zorder = 1)
gl.top_labels = False
gl.right_labels = False
gl.xlines = True
gl.ylines = True
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()



centroids = np.array(stats[stats.id == 1].centroid.iloc[0])
lons_centr, lats_centr = zip(*centroids)
stat_time = pd.to_datetime(np.array(stats[stats.id == 1].dates.iloc[0]))

# fig, ax = plt.subplots(figsize=(10, 10),dpi = 300, subplot_kw={'projection': ccrs.PlateCarree()})

# # Add a map feature
# ax.coastlines()
# ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
import matplotlib.dates as mdates
# Plot the points
ax.plot(lons_centr, lats_centr,   transform=ccrs.PlateCarree(),
           color = 'black')

stat_time_num = mdates.date2num(stat_time)
scatter = ax.scatter(lons_centr, lats_centr,  marker='o', transform=ccrs.PlateCarree(),
           c = stat_time_num, cmap = 'jet', edgecolor = 'black',zorder = 4)
# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal')  # 'orientation' can be 'vertical' or 'horizontal'
# Set the colorbar ticks and labels to datetime format
cbar.set_ticks(np.linspace(stat_time_num.min(), stat_time_num.max(), num=5))
cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# Optionally label the colorbar
cbar.set_label('Time of center of MHW')
# # Set title and labels
# ax.set_title('Longitude and Latitude Plot')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_ylim([-35,-10])
# ax.set_xlim([100,117.5])
plt.show()




#%%
blobs.attrs



lat_watr = -31.6
lon_watr  = 115.0733 

maxl = int(np.nanmax(blobs.values)) + 1
Nr_blobs = blobs.attrs['final objects tracked']
base_cmap = plt.get_cmap('gist_rainbow')
colors = base_cmap(np.linspace(0, 1, Nr_blobs))
# shuffled_arr = np.random.permutation(colors)
custom_cmap = ListedColormap(colors)



# cm = ListedColormap(np.random.random(size=(maxl, 3)).tolist())
lon = hot_water_short.longitude.data
lat = hot_water_short.latitude.data
# t = 6500,

filenames = []
f = 0
ii = 0


for id_t,t in enumerate(range(20)): #len(hot_water_short.time.data))): #len(hot_water_short.time.data)): #
    
    fig, axs = plt.subplots(ncols=2, nrows =1, figsize=(7,5),dpi = 200,
                            sharex=True,sharey=True)
    
    fig.suptitle(str(hot_water_short.time[t].data)[:10] + ' - depth: 0m')
      
    fig.subplots_adjust(wspace=0.7)
        
    levels = np.linspace(1,Nr_blobs,Nr_blobs )
    
    ax = axs[1]
    ax.contourf(lon,lat,mask2.thetao.where(mask2.thetao==0),colors='grey') 
    ax.contourf(lon,lat,mask_surface.thetao.where(mask_surface.thetao==0),colors='k') 
    
    cm = ax.contourf(lon,lat,blobs.isel(time=t).data,Nr_blobs, cmap = custom_cmap, #ArithmeticErrovmin=0.5, vmax=Nr_blobs+0.5,
                    levels = np.linspace(1,Nr_blobs+1,Nr_blobs + 1)
                    )
    cbaxes = fig.add_axes([0.95, 0.2, 0.035, 0.6])
    cb = fig.colorbar(cm, orientation='vertical',cax=cbaxes,aspect = 0.5,shrink=0.8, ax = ax, 
                       ticks = levels + 0.5
                      )
    cb.set_label('Label', size=10)
    cb.ax.tick_params(labelsize=9)
    cb.set_ticks(levels + 0.5)
    cb.set_ticklabels(levels)    
    
    ax.scatter(lon_watr,lat_watr, edgecolor = 'white',facecolor='navy', s = 50)    
        
        
    ax.set_aspect('equal')
    
    ax = axs[0]
    
    cm = ax.contourf(lon,lat,hot_water_short.thetao.isel(time=t).data,50, cmap='Reds', 
                     levels = np.linspace(0,5,50), extend='max')
    
    cbaxes = fig.add_axes([0.46, 0.2, 0.025, 0.6])
    cb = fig.colorbar(cm, orientation='vertical',cax=cbaxes,aspect = 0.5,shrink=0.8, 
                      ax = ax, ticks = np.linspace(0,5,6))
    cb.set_label('Intensity [°C]', size=10)
    cb.ax.tick_params(labelsize=9)
    ax.contourf(lon,lat,mask2.thetao.where(mask2.thetao==0),colors='grey') 
    ax.contourf(lon,lat,mask_surface.thetao.where(mask_surface.thetao==0),colors='k') 
    ax.set_aspect('equal');
    ax.scatter(lon_watr,lat_watr, edgecolor = 'white',facecolor='navy', s = 50)   
    
    

    ax.set_xticks(lon[::6])
    


#     filename = f'{ii}.png'
#     ii = ii + 1
#     f = f + 1
#     filenames.append(filename)

#     # save frame
#     plt.savefig(filename, bbox_inches = 'tight')
#     plt.close()
     

# filenames.append(filenames[-1])
# clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(filenames, fps=3.)
# clip.write_videofile('C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/Ocetrac/Ocetrac3D_WA_depth' +str(d_sel) +'.mp4')     
        
# # Remove files
# for filename in set(filenames):
#     os.remove(filename)




#%%


maxl = int(np.nanmax(blobs.values)) + 1
Nr_blobs = blobs.attrs['final objects tracked']
base_cmap = plt.get_cmap('gist_rainbow')
colors = base_cmap(np.linspace(0, 1, Nr_blobs))
custom_cmap = ListedColormap(colors)


maxl = int(np.nanmax(blobs50.values)) + 1
Nr_blobs50 = blobs50.attrs['final objects tracked']
base_cmap = plt.get_cmap('gist_rainbow')
colors = base_cmap(np.linspace(0, 1, Nr_blobs50))
custom_cmap50 = ListedColormap(colors)
    
maxl = int(np.nanmax(blobs100.values)) + 1
Nr_blobs100 = blobs100.attrs['final objects tracked']
base_cmap = plt.get_cmap('gist_rainbow')
colors = base_cmap(np.linspace(0, 1, Nr_blobs100))
custom_cmap100 = ListedColormap(colors)  

# cm = ListedColormap(np.random.random(size=(maxl, 3)).tolist())
lon = hot_water_short.longitude.data
lat = hot_water_short.latitude.data
# t = 6500,

filenames = []
f = 0
ii = 0


for id_t,t in enumerate(range(len(hot_water_short.time.data))): #len(hot_water_short.time.data)): #2)): #
    
    fig, axs = plt.subplots(ncols=2, nrows =3, figsize=(7,15),dpi = 200,
                            sharex=True,sharey=True)
    
    fig.suptitle(str(hot_water_short.time[t].data)[:10] ) #+ ' - depth: ' + str(d_sel) +'m')
    fig.subplots_adjust(wspace=0.7, hspace=0.03)
    plt.subplots_adjust(top=.95)    
    
    

    
    
    #### blob at surface
    levels = np.linspace(1,Nr_blobs,Nr_blobs )
    ax = axs[0,1]
    ax_pos = ax.get_position()
    
    ax.contourf(lon,lat,mask2.thetao.where(mask2.thetao==0),colors='grey') 
    ax.contourf(lon,lat,mask_surface.thetao.where(mask_surface.thetao==0),colors='k') 
    
    cm = ax.contourf(lon,lat,blobs.isel(time=t).data, cmap = custom_cmap, #vmin=1, vmax=Nr_blobs
                    levels = np.linspace(1,Nr_blobs+1,Nr_blobs + 1)
                    )
    cbaxes = fig.add_axes([0.95, ax_pos.y0, 0.035, ax_pos.y1 - ax_pos.y0])
    cb = fig.colorbar(cm, orientation='vertical',cax=cbaxes,aspect = 0.5,shrink=0.8, ax = ax, 
                      ticks = levels + 0.5)
    cb.set_label('Label', size=10)
    cb.ax.tick_params(labelsize=9)
    cb.set_ticks(levels + 0.5)
    cb.set_ticklabels(levels)    
    ax.scatter(lon_watr,lat_watr, edgecolor = 'white',facecolor='navy', s = 50)    
    ax.set_aspect('equal')
    
    
    #### blob 50
    levels = np.linspace(1,Nr_blobs50,Nr_blobs50 )
    ax = axs[1,1]
    ax_pos = ax.get_position()
    ax.contourf(lon,lat,mask50.thetao.where(mask50.thetao==0),colors='grey') 
    ax.contourf(lon,lat,mask_surface.thetao.where(mask_surface.thetao==0),colors='k') 
    
    cm = ax.contourf(lon,lat,blobs50.isel(time=t).data, cmap = custom_cmap50, #vmin=1, vmax=Nr_blobs
                    levels = np.linspace(1,Nr_blobs50+1,Nr_blobs50 + 1)
                    )
    cbaxes = fig.add_axes([0.95, ax_pos.y0, 0.035, ax_pos.y1 - ax_pos.y0])
    cb = fig.colorbar(cm, orientation='vertical',cax=cbaxes,aspect = 0.5,shrink=0.8, ax = ax, 
                      ticks = levels + 0.5)
    cb.set_label('Label', size=10)
    cb.ax.tick_params(labelsize=9)
    cb.set_ticks(levels + 0.5)
    cb.set_ticklabels(levels)    
    ax.scatter(lon_watr,lat_watr, edgecolor = 'white',facecolor='navy', s = 50)    
    ax.set_aspect('equal')
    
    
    #### blob at 100m
    levels = np.linspace(1,Nr_blobs100,Nr_blobs100 )
    ax = axs[2,1]
    ax_pos = ax.get_position()
    
    ax.contourf(lon,lat,mask100.thetao.where(mask100.thetao==0),colors='grey') 
    ax.contourf(lon,lat,mask_surface.thetao.where(mask_surface.thetao==0),colors='k') 
    
    cm = ax.contourf(lon,lat,blobs100.isel(time=t).data, cmap = custom_cmap100, #vmin=1, vmax=Nr_blobs
                    levels = np.linspace(1,Nr_blobs100+1,Nr_blobs100 + 1)
                    )
    cbaxes = fig.add_axes([0.95, ax_pos.y0, 0.035, ax_pos.y1 - ax_pos.y0])
    cb = fig.colorbar(cm, orientation='vertical',cax=cbaxes,aspect = 0.5,shrink=0.8, ax = ax, 
                      ticks = levels + 0.5)
    cb.set_label('Label', size=10)
    cb.ax.tick_params(labelsize=9)
    cb.set_ticks(levels + 0.5)
    cb.set_ticklabels(levels)    
    ax.scatter(lon_watr,lat_watr, edgecolor = 'white',facecolor='navy', s = 50)    
    ax.set_aspect('equal')
    
    
    
    ## hot water at surface 
    ax = axs[0,0]
    cm = ax.contourf(lon,lat,hot_water_short.thetao.isel(time=t).data,50, cmap='Reds', 
                     levels = np.linspace(0,5,50), extend='max')
    
    # cbaxes = fig.add_axes([0.46, 0.2, 0.025, 0.6])
    # cb = fig.colorbar(cm, orientation='vertical',cax=cbaxes,aspect = 0.5,shrink=0.8, 
    #                   ax = ax, ticks = np.linspace(0,5,6))
    # cb.set_label('Intensity [°C]', size=10)
    # cb.ax.tick_params(labelsize=9)
    ax.contourf(lon,lat,mask2.thetao.where(mask2.thetao==0),colors='grey') 
    ax.contourf(lon,lat,mask_surface.thetao.where(mask_surface.thetao==0),colors='k') 
    ax.set_aspect('equal');
    ax.scatter(lon_watr,lat_watr, edgecolor = 'white',facecolor='navy', s = 50)   
    ax.set_xticks(lon[::6])
    ax.set_ylabel('0m')
    
    ## hot water at surface 
    ax = axs[1,0]
    cm = ax.contourf(lon,lat,hot_water_short50.thetao.isel(time=t).data,50, cmap='Reds', 
                     levels = np.linspace(0,5,50), extend='max')
    
    cbaxes = fig.add_axes([0.46, 0.38, 0.025, 0.3])
    cb = fig.colorbar(cm, orientation='vertical',cax=cbaxes,aspect = 0.5,shrink=0.8, 
                      ax = ax, ticks = np.linspace(0,5,6))
    cb.set_label('Intensity [°C]', size=10)
    cb.ax.tick_params(labelsize=9)
    ax.contourf(lon,lat,mask50.thetao.where(mask50.thetao==0),colors='grey') 
    ax.contourf(lon,lat,mask_surface.thetao.where(mask_surface.thetao==0),colors='k') 
    ax.set_aspect('equal');
    ax.scatter(lon_watr,lat_watr, edgecolor = 'white',facecolor='navy', s = 50)   
    ax.set_xticks(lon[::6])
    ax.set_ylabel('50m')
    
    ## hot water at surface 
    ax = axs[2,0]
    cm = ax.contourf(lon,lat,hot_water_short100.thetao.isel(time=t).data,50, cmap='Reds', 
                     levels = np.linspace(0,5,50), extend='max')
    
    # cbaxes = fig.add_axes([0.46, 0.2, 0.025, 0.6])
    # cb = fig.colorbar(cm, orientation='vertical',cax=cbaxes,aspect = 0.5,shrink=0.8, 
    #                   ax = ax, ticks = np.linspace(0,5,6))
    # cb.set_label('Intensity [°C]', size=10)
    # cb.ax.tick_params(labelsize=9)
    ax.contourf(lon,lat,mask100.thetao.where(mask100.thetao==0),colors='grey') 
    ax.contourf(lon,lat,mask_surface.thetao.where(mask_surface.thetao==0),colors='k') 
    ax.set_aspect('equal');
    ax.scatter(lon_watr,lat_watr, edgecolor = 'white',facecolor='navy', s = 50)   
    ax.set_xticks(lon[::6])
    ax.set_ylabel('100m')
    
    


    filename = f'{ii}.png'
    ii = ii + 1
    f = f + 1
    filenames.append(filename)

    # save frame
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close()
     

filenames.append(filenames[-1])
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(filenames, fps=3.)
clip.write_videofile('C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/Ocetrac/Ocetrac3D_WA_depth0-100.mp4')     
        
# Remove files
for filename in set(filenames):
    os.remove(filename)



#%%


d_sel = 0
hot_water_short = hot_water.sel(time = slice('2010-11-01','2011-05-01'))#.sel( depth = d_sel, method = 'nearest')
# mask2 = mask.sel( depth = d_sel, method = 'nearest')
mask_surface = mask.sel( depth = 0, method = 'nearest')

blobs_all = []

for d_sel in hot_water.depth.data:
    
    Tracker = Tracker3D(hot_water_short.thetao.sel(depth=d_sel), mask.thetao.sel(depth=d_sel), 
                        radius=2, min_size_quartile=0.75,
                        timedim = 'time', xdim = 'longitude', ydim='latitude',
                        positive=True)
    blobs = Tracker.track()
    
    # Expand the depth dimension and assign the current depth value
    blobs_depth = blobs.expand_dims('depth').assign_coords(depth=[d_sel])

    # Append to the list
    blobs_all.append(blobs_depth)

# Concatenate all xarrays along the depth dimension
merged_blobs = xr.concat(blobs_all, dim='depth')


blobs_dataset = xr.Dataset({'Blobs': merged_blobs})

folder_saving = 'C:/Users/24048369/OneDrive - The University of Western Australia/MHW_Project_FHL/Ocetrac/'
blobs_dataset.to_netcdf(folder_saving + 'merged_blobs_2010-2011.nc')


mask
mask.to_netcdf(folder_saving + 'mask_2010-2011.nc')
#%%











# blobs.attrs




# maxl = int(np.nanmax(blobs.values)) + 1
# Nr_blobs = blobs.attrs['final objects tracked']
# base_cmap = plt.get_cmap('gist_rainbow')
# colors = base_cmap(np.linspace(0, 1, Nr_blobs+1))
# custom_cmap = ListedColormap(colors)


# # np.random.seed(42)  # For reproducibility
# # colors = np.random.rand(Nr_blobs+1, 3)  # Generate 13 random RGB colors
# # # Step 2: Create a custom colormap
# # custom_cmap = mcolors.ListedColormap(colors)


# # cmap_rainbow = plt.cm.gist_rainbow
# # colors = cmap_rainbow(np.linspace(0, 1, Nr_blobs+1))
# # np.random.seed(42)  
# # np.random.shuffle(colors)
# # custom_cmap = mcolors.ListedColormap(colors)



# cm = ListedColormap(np.random.random(size=(maxl, 3)).tolist())
# lon = hot_water_short.longitude.data
# lat = hot_water_short.latitude.data
# t = 8500,
# for id_t,t in enumerate(range(65,72)): #len(hot_water_short.time.data)):
    
#     fig, axs = plt.subplots(ncols=2, nrows =6, figsize=(6,10),dpi = 100,
#                             sharex=True,sharey=True)
    
#     fig.subplots_adjust(hspace=0)
    
#     for id_d,d in enumerate([0,10,20,30,50,100]):
        
        
#         ax = axs[id_d,1]
#         if id_d ==0:
#             ax.set_title(str(hot_water_short.time[t].data)[:10])
        
#         ax.set_ylabel('depth:' + str(np.round(hot_water_short.sel(depth = d, method = 'nearest').depth.data)))
        
#         levels = np.linspace(0,Nr_blobs,Nr_blobs + 1)
#         ax.contourf(lon,lat,mask2.thetao.sel(depth = d, method ='nearest').where(mask2.thetao.sel(depth = d, method = 'nearest')==0),colors='k') 
        
#         cm = ax.contourf(lon,lat,blobs.isel(time=t, depth=id_d).data, cmap = custom_cmap, #vmin=1, vmax=Nr_blobs
#                         levels = np.linspace(0,Nr_blobs+1,Nr_blobs + 2)
                        
#                         )
        
        
        
        
#         ax.set_aspect('equal')
#         ax = axs[id_d,0]
        
#         ax.contourf(lon,lat,hot_water_short.thetao.isel(time=t,depth = id_d).data, cmap = 'Reds')
#         ax.contourf(lon,lat,mask2.thetao.sel(depth = d, method = 'nearest').where(mask2.thetao.sel(depth = d, method = 'nearest')==0),colors='k') 
#         ax.set_aspect('equal');
    
    
#     cbaxes = fig.add_axes([0.95, 0.2, 0.035, 0.6])
    
#     cb = fig.colorbar(cm, orientation='vertical',cax=cbaxes,aspect = 0.5,shrink=0.8, ax = ax1, 
#                       ticks = levels + 0.5)
#     cb.set_label('Label', size=12)
#     cb.ax.tick_params(labelsize=6)
#     cb.set_ticks(levels + 0.5)
#     cb.set_ticklabels(levels)

#     plt.savefig(folder + str(id_t) +'_depth_' +str(hot_water_short.time[t].data)[:10] + '.png', bbox_inches='tight',dpi = 300)
#     # plt.savefig(folder + str(id_t) +'_depth_' +str(hot_water_short.time[t].data)[:10] + '.pdf', bbox_inches='tight',dpi = 300)






#%%


fig, ax = plt.subplots(figsize=(10, 10),dpi = 300, subplot_kw={'projection': ccrs.PlateCarree()})

# Add a map feature
ax.coastlines()
# ax.add_feature(cartopy.feature.BORDERS, linestyle=':')

# Plot the points
# ax.scatter(lons_centr, lats_centr, color='red', marker='o', transform=ccrs.PlateCarree())

# # Set title and labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

ax.grid()
ax.set_xlim([109,116])
# # ax.set_zlim([-210,10])
ax.set_ylim([-35.5,-26.5])



x1 = 113.5
x2 = 115.75
y1 = -31.2500
y2 = -32.0000

x = [x2, x2, x1, x1, x2]
y = [y1, y2, y2, y1, y1]

ax.plot(x, y, transform=ccrs.PlateCarree(), color = 'turquoise', zorder = 3)
 
ax.scatter(lon_watr,lat_watr)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.8, color='gray', alpha=0.7, linestyle='--', zorder = 1)
gl.top_labels = False
# gl.left_labels = False
gl.right_labels = False
# gl.bottom_labels = False
gl.xlines = True
gl.ylines = True
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()


















