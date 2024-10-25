# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:19:43 2024

@author: 24048369
"""
import copy
import xarray as xr
import numpy as np
import pandas as pd
import scipy.ndimage
from skimage.measure import regionprops 
from skimage.measure import label as label_np
import dask.array as dsa
import numpy as np
from scipy.ndimage import label, find_objects
from sklearn.metrics.pairwise import haversine_distances


def _apply_mask(binary_images, mask):
    binary_images_with_mask = binary_images.where(mask==1, drop=False, other=0)
    return binary_images_with_mask

class Tracker4D:
        
    def __init__(self, da, mask, radius, min_size_quartile, timedim, xdim, ydim, ddim, positive=True):
        
        

        self.da = da
        self.mask = mask
        self.radius = radius
        self.min_size_quartile = min_size_quartile
        self.timedim = timedim
        self.xdim = xdim
        self.ydim = ydim   
        self.ddim = ddim  
        self.positive = positive
        
        if ((timedim, ydim, xdim, ddim) != da.dims):
            try:
                da = da.transpose(timedim, ydim, xdim,ddim) 
            except:
                raise ValueError(f'Ocetrac3D currently only supports 4D DataArrays (time, lat,lon,depth). The dimensions should only contain ({timedim}, {xdim}, {ydim}, and {ddim}). Found {list(da.dims)}')

            
    def track(self):
        '''
        Label and track image features.
        
        Parameters
        ----------
        da : xarray.DataArray
            The data to label.

        mask : xarray.DataArray
            The mask of ponts to ignore. Must be binary where 1 = true point and 0 = background to be ignored. 

        radius : int
            The size of the structuring element used in morphological opening and closing. Radius specified by the number of grid units.

        min_size_quartile : float
            The quantile used to define the threshold of the smallest area object retained in tracking. Value should be between 0 and 1.

        timedim : str
            The name of the time dimension
        
        xdim : str
            The name of the x dimension

        ydim : str
            The namne of the y dimension
        
        ddim : str
            The namne of the depth dimension
            
        positive : bool
            True if da values are expected to be positive, false if they are negative. Default argument is True

        Returns
        -------
        labels : xarray.DataArray
            Integer labels of the connected regions.
        '''

        if (self.mask == 0).all():
            raise ValueError('Found only zeros in `mask` input. The mask should indicate valid regions with values of 1')

        # Convert data to binary, define structuring element, and perform morphological closing then opening
        binary_images = self._morphological_operations()

        # Apply mask
        binary_images_with_mask  = _apply_mask(binary_images,self.mask) # perhaps change to method? JB

        # Filter area
        area, min_area, binary_labels, N_initial = self._filter_area(binary_images_with_mask)


        # Label objects
        labels, num = self._label_either(binary_labels, return_num= True, connectivity=3)

        # Wrap labels
        grid_res = abs(self.da[self.xdim][1]-self.da[self.xdim][0])
        if self.da[self.xdim][-1]-self.da[self.xdim][0] >= 360-grid_res:
            labels_wrapped, N_final = self._wrap(labels)
        else:
            labels_wrapped = labels
            N_final = np.max(labels)
                

        
        
        labels_wrapped = np.transpose(labels_wrapped, (0, 3, 1, 2))
        
        new_labels = xr.DataArray(labels_wrapped, dims=self.da.dims, coords=self.da.coords)   
        new_labels = new_labels.where(new_labels!=0, drop=False, other=np.nan)


        ## Metadata

        # Calculate Percent of total object area retained after size filtering
        sum_tot_area = int(np.sum(area.values))

        reject_area = area.where(area<=min_area, drop=True)
        sum_reject_area = int(np.sum(reject_area.values))
        percent_area_reject = (sum_reject_area/sum_tot_area)

        accept_area = area.where(area>min_area, drop=True)
        sum_accept_area = int(np.sum(accept_area.values))
        percent_area_accept = (sum_accept_area/sum_tot_area)

        new_labels = new_labels.rename('labels')
        new_labels.attrs['inital objects identified'] = int(N_initial)
        new_labels.attrs['final objects tracked'] = int(N_final)
        new_labels.attrs['radius'] = self.radius
        new_labels.attrs['size quantile threshold'] = self.min_size_quartile
        new_labels.attrs['min area'] = min_area
        new_labels.attrs['percent area reject'] = percent_area_reject
        new_labels.attrs['percent area accept'] = percent_area_accept

        print('inital objects identified \t', int(N_initial))
        print('final objects tracked \t', int(N_final))

        return new_labels


    ### PRIVATE METHODS - not meant to be called by user ###
    def collect_volume_stats(self, labels):
        """Collects surface statistics"""
    
        ids = np.unique(labels)
        ids = np.array([id for id in ids if ~np.isnan(id)])
        
        print(ids)
        
        dataframes = []
        num_events = len(ids)
    
        for i in range(num_events):
            event = labels.where(labels == ids[i], drop=True)
            df = self._get_volume_stats(event) #self.da or sst
            dataframes.append(df)
            #break
    
        dff = pd.concat(dataframes, ignore_index=True)
        
        df_cleaned = dff.dropna(subset=['id']).reset_index(drop=True)
        
        
        return df_cleaned

    ### PRIVATE METHODS - not meant to be called by user ###

    def _compute_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
        """Compute the x and y resolution in km for a geographic degree resolution
    
        :param clat: the point latitude at which to compute the resolution
        :param clon: the point longitude at which to compute the resolution
        :param res: the geographic degree resolution
        :return: distance in km between two points
        """
        
        dist = haversine_distances(
            np.deg2rad(np.array([(lat1, lon1)])),
            np.deg2rad(np.array([(lat2, lon2)])),
        )
        
        dist = dist * 6371000 / 1000  # multiply by Earth radius to get kilometers
        dist
        return dist[0][0]
    
    
    # def calculate_object_volume(data, object_id):
    #     """
    #     Calculate the volume of an object defined by a specific integer ID in a 3D xarray with non-uniform depth spacing.
    
    #     Parameters:
    #     - data: xarray.DataArray with dimensions (longitude, latitude, depth)
    #     - object_id: Integer defining the object of interest
    
    #     Returns:
    #     - Volume of the object
    #     """
    #     # Mask the data to isolate the object
    #     object_mask = (data == object_id)
         
    #     R = 6371
    #     lat2d, lon2d = np.meshgrid(data.latitude.data, data.longitude.data, indexing='ij')
    #     lat2d_radians = np.deg2rad(lat2d)
    #     lon2d_radians = np.deg2rad(lon2d)

    #     dy = R * (lat2d_radians[1:, :] - lat2d_radians[:-1, :])
    #     lat_spacing_km = np.vstack((np.zeros((1, dy.shape[1])), dy))
    #     lon_spacing_km = R * np.cos(lat2d_radians) * (lon2d_radians[:, 1:] - lon2d_radians[:, :-1])[0,0]
        
    #     # Initialize volume array
    #     volume = np.zeros(data.shape, dtype=float)
        
    #     # Iterate over each depth level to calculate cell volumes
    #     for i in range(len(data.depth) - 1):
    #         # Compute depth spacing for this level
    #         depth_spacing = data.depth[i + 1] - data.depth[i]
    #         # Calculate volume for this depth level
    #         cell_volume = lon_spacing_km[:, np.newaxis] * lat_spacing_km[np.newaxis, :] * depth_spacing
    #         volume[:, :, i] = cell_volume
        
    #     # Mask the cell volumes to only include those that belong to the object
    #     object_volume = volume * object_mask
        
    #     # Sum the volumes of the cells that belong to the object
    #     total_volume = np.sum(object_volume)
        
    #     return total_volume    
    
    
    
    

    def _get_volume_stats(self, event):
        """ get surface stats for one event """
        
        # Initialize dictionary 
        mhw = {}
        mhw['id'] = [] # event label
        mhw['dates'] = [] # datetime format
        mhw['coords'] = [] # (lat, lon,depth)
        mhw['centroid'] = []  # (lat, lon,depth)
        mhw['duration'] = [] # [time]
        mhw['intensity_max'] = [] # [deg C]
        mhw['intensity_max_depthlevel'] = [] # [m]
        mhw['intensity_mean'] = [] # [deg C]
        mhw['intensity_min'] = [] # [deg C]
        mhw['intensity_cumulative'] = [] # [deg C]
        mhw['total_area_surface'] = [] # [km2]
        mhw['surface_area_change'] = [] # [km2]
        mhw['volume_change'] = [] # [km^3]
        mhw['max_depth'] = [] # [m]
        mhw['mean_depth'] = [] # [m]
        mhw['distance'] = [] 
        
        # TO ADD:
        # mhw['rate_onset'] = [] # [deg C / month]
        # mhw['rate_decline'] = [] # [deg C / month]
        
        mhw["id"].append(int(np.nanmedian(event.values)))
        mhw["dates"].append(event.time.values)
        mhw["duration"] = event.time.shape[0]
        mhw["total_area_surface"] = int(event.sel(depth = 0,method = 'nearest').sum("time").count((self.ydim,self.xdim )).values)


        # Process intensity metrics using try-except to handle ValueError
        event_ssta = self.da.where(event > 0, drop=True)

            
            
        volume_list   = []
        centroid_list = []
        distance_list = []
        surface_area_changee_list = []
        
        max_int_depthlevels_list = []
        
        prev_coords = None  # Initialize previous coordinates as None
        
        for time in event.time.values:
            mhw_slice = event.sel(time=time)
            
            event_ssta_slice = event_ssta.sel(time=time)
            
            mhw_slice_surf = mhw_slice.sel(depth = 0,method = 'nearest')
            surf_area = int(mhw_slice_surf.count((self.ydim,self.xdim )).values)
            surface_area_changee_list.append(surf_area)
            
            
            # Create latitude and longitude grids
            lon_grid, lat_grid = np.meshgrid(np.arange(0, mhw_slice[self.xdim].shape[0]), np.arange(0, mhw_slice[self.ydim].shape[0]))
            lon_flat = lon_grid.flatten()
            lat_flat = lat_grid.flatten()
            
            
            #### find the depth with most integers
            integer_counts = []
            # Iterate through each depth level
            for depth in mhw_slice.depth.values:
                # Extract the data for the current depth level
                depth_data = mhw_slice.sel(depth=depth)
                # Count non-zero integers (assuming zeros are not considered)
                count_integers = np.nansum(depth_data.values != 0)  # Change the condition if needed
                integer_counts.append(count_integers)
            # Find the index of the maximum count
            max_index = np.argmax(integer_counts)

            if np.size(max_index) >1:
                max_index = max_index[0]
                
                
                
            
            ## select the depth with the biggest area = most integer count
            data_flat = mhw_slice.isel(depth = max_index).values.flatten()
            
            # Compute the centroid
            centroid_lon = np.nansum(lon_flat * data_flat) / np.nansum(data_flat)
            centroid_lat = np.nansum(lat_flat * data_flat) / np.nansum(data_flat)

           
            if isinstance(centroid_lat, (int, float, complex)) and not isinstance(centroid_lat, bool) and not (isinstance(centroid_lat, float) and np.isnan(centroid_lat)):
                
                
                coords = [(mhw_slice[self.xdim][round(centroid_lon)].values), (mhw_slice[self.ydim][round(centroid_lat)].values)]
                centroid_list.append(coords)
                lon2, lat2 = coords
                if prev_coords is not None and np.isnan(prev_coords[0]) == False:
                    lon1, lat1 = prev_coords  
                    distance = self._compute_distance(lat1, lon1, lat2, lon2)
                    distance_list.append(distance)
                
            else:
                # print(centroid_lat,centroid_lon)
                coords = [np.nan, np.nan]
                centroid_list.append([np.nan,np.nan])
                distance_list.append(np.nan)
            
         
                
            prev_coords = coords  
            
            
            ####### calculate volume for each timestep 
            # Mask the data to isolate the object
            object_mask = (mhw_slice == int(np.nanmedian(event.values)))
             
            R = 6371
            lat2d, lon2d = lat_grid,lon_grid
            
            lat2d_radians = np.deg2rad(lat2d)
            lon2d_radians = np.deg2rad(lon2d)
    
            dy = R * (lat2d_radians[1:, :] - lat2d_radians[:-1, :])
            dy = np.vstack((np.zeros((1, dy.shape[1])), dy))
            dx = R * np.cos(lat2d_radians) * (lon2d_radians[:, 1:] - lon2d_radians[:, :-1])[0,0]
            lon_spacing_km = dx[0,:]
            lat_spacing_km = dy[:,1]
           
            
            # print(len(lat_spacing_km))
            # print(len(lon_spacing_km))
            # Initialize volume array
            volume = np.zeros(mhw_slice.shape, dtype=float)
            
            # print('size volume: ' + str(np.shape(volume)))
            # print('size mhw_slice: ' + str(np.shape(mhw_slice)))
            # Iterate over each depth level to calculate cell volumes
            for i in range(len(mhw_slice.depth) - 1):
                # Compute depth spacing for this level
                depth_spacing = (mhw_slice.depth.data[i + 1] - mhw_slice.depth.data[i])/1000
                # Calculate volume for this depth level
                cell_volume = lon_spacing_km[:, np.newaxis] * lat_spacing_km[np.newaxis, :] * depth_spacing
                volume[i, :, :] = cell_volume.T
            
            # Mask the cell volumes to only include those that belong to the object
            object_volume = volume * object_mask
            
            # Sum the volumes of the cells that belong to the object
            total_volume = np.sum(object_volume)
            
         
            
            volume_list.append(total_volume.data)
            #######
            
            
            
            
            # ### calculate depthlevel of maximum intensity at each timestep 
            id1,id2,id3 = np.where(event_ssta_slice.values == np.nanmax(event_ssta_slice.data))
            max_int_depthlevels_list.append(event_ssta_slice.depth.values[id1])
            
                
            
            
        
        # Add a None or NaN for the first distance since it's undefined
        distance_list.insert(0, np.nan)
        #distance_list.insert(0, None)  # or use `np.nan` if you prefer
        
        mhw['centroid'].append(centroid_list)
        mhw['distance'].append(distance_list)
        mhw['volume_change'].append(volume_list)
        mhw['surface_area_change'].append(surface_area_changee_list)
        mhw['intensity_max_depthlevel'].append(max_int_depthlevels_list)
       
        
        try:
            mhw['intensity_mean'].append(event_ssta.mean((self.ydim, self.xdim,self.ddim)).values)
        except ValueError:
            mhw['intensity_mean'].append(np.nan)
        try:
            mhw['intensity_max'].append(event_ssta.max((self.ydim, self.xdim,self.ddim)).values)
            
        except ValueError:
            mhw['intensity_max'].append(np.nan)
        try:
            mhw['intensity_min'].append(event_ssta.min((self.ydim, self.xdim,self.ddim)).values)
        except ValueError:
            mhw['intensity_min'].append(np.nan)
        try:
            mhw['intensity_cumulative'].append(np.nansum(event_ssta))
        except ValueError:
            mhw['intensity_cumulative'].append(np.nan)
        
            
        coords = event.stack(z=(self.ydim, self.xdim))
        coord_pairs = [(coords.isel(time=t[0]).dropna(dim='z', how='any').z[self.ydim].values, 
                          coords.isel(time=t[0]).dropna(dim='z', how='any').z[self.xdim].values) for t in enumerate(event.time)]
        
        mhw['coords'].append(coord_pairs)
        
        
        
        mhw = pd.DataFrame(dict([(name, pd.Series(data)) for name,data in mhw.items()]))
        return mhw   
    
    
    




    def _morphological_operations(self): 
        '''Converts xarray.DataArray to binary, defines structuring element, and performs morphological closing then opening.
        Parameters
        ----------
        da     : xarray.DataArray
                The data to label
        radius : int
                Length of grid spacing to define the radius of the structing element used in morphological closing and opening.

        '''

        # Convert images to binary. All positive values == 1, otherwise == 0
        if self.positive == True:
            bitmap_binary = self.da.where(self.da>0, drop=False, other=0)
        
        elif self.positive == False:
            bitmap_binary = self.da.where(self.da<0, drop=False, other=0)
    
        bitmap_binary = bitmap_binary.where(bitmap_binary==0, drop=False, other=1)
        
        # Define structuring element
        diameter = self.radius*2
        x = np.arange(-self.radius, self.radius+1)
        x, y, z = np.meshgrid(x, x, x)
        
        r = x**2 + y**2 + z**2
        se = r<self.radius**2
        def binary_open_close(bitmap_binary):
            bitmap_binary_padded = np.pad(bitmap_binary,diameter,
                                         
                                          mode='wrap')
            
            
            s1 = scipy.ndimage.binary_closing(bitmap_binary_padded, se, iterations=1)
            s2 = scipy.ndimage.binary_opening(s1, se, iterations=1)
            unpadded= s2[diameter:-diameter, diameter:-diameter,diameter:-diameter]
            return unpadded

        mo_binary = xr.apply_ufunc(binary_open_close, bitmap_binary,
                                   input_core_dims=[[self.ydim, self.xdim,self.ddim]],
                                   output_core_dims=[[self.ydim, self.xdim, self.ddim]],
                                   output_dtypes=[bitmap_binary.dtype],
                                   vectorize=True,
                                   dask='parallelized')
        return mo_binary


    def _filter_area(self, binary_images):
        '''calculatre area with regionprops'''

        def get_labels(binary_images):
            blobs_labels = self._label_either(binary_images, background=0)
            return blobs_labels

        labels = xr.apply_ufunc(get_labels, binary_images,
                                input_core_dims=[[self.ydim, self.xdim, self.ddim]],
                                output_core_dims=[[self.ydim, self.xdim, self.ddim]],
                                output_dtypes=[binary_images.dtype],
                                vectorize=True,
                                dask='parallelized')


        labels = xr.DataArray(labels, dims=binary_images.dims, coords=binary_images.coords)
        labels = labels.where(labels>0, drop=False, other=np.nan)  
        # The labels are repeated each time step, therefore we relabel them to be consecutive
        for i in range(1, labels.shape[0]):
            labels[i,:,:] = labels[i,:,:].values + labels[i-1,:,:].max().values

        labels = labels.where(labels>0, drop=False, other=0)  
        labels_wrapped, N_initial = self._wrap(np.array(labels))
        
        
               
        ### get just surface area
        props_surface = regionprops(labels_wrapped[:,:,:,0].astype('int'))
        labelprops = [p.label for p in props_surface]
        labelprops = xr.DataArray(labelprops, dims=['label'], coords={'label': labelprops}) 
        area = xr.DataArray([p.area for p in props_surface], dims=['label'], coords={'label': labelprops})  # Number of pixels of the region.
        
        if area.size == 0:
            raise ValueError(f'No objects were detected. Try changing radius or min_size_quartile parameters.')
        
        min_area = np.percentile(area, self.min_size_quartile*100)
        
        keep_labels = labelprops.where(area>=min_area, drop=True)
        keep_where = np.isin(labels_wrapped[:,:,:,0], keep_labels)
        out_labels = xr.DataArray(np.where(keep_where==False, 0, labels_wrapped[:,:,:,0]), 
                                  dims=binary_images.sel(depth = 0,method = 'nearest').drop('depth').dims, 
                                  coords=binary_images.sel(depth = 0,method = 'nearest').drop('depth').coords)

        # # Convert images to binary. All positive values == 1, otherwise == 0
        binary_labels = out_labels.where(out_labels==0, drop=False, other=1)
        
        
        binary_labels = copy.deepcopy(binary_images)
        depth_levels = binary_labels.depth.data
        
        for each_depth in range( np.shape(labels_wrapped)[3]) :
            
            
            
            props = regionprops(labels_wrapped[:,:,:,each_depth].astype('int'))
            labelprops = [p.label for p in props]
            labelprops = xr.DataArray(labelprops, dims=['label'], coords={'label': labelprops}) 
            area = xr.DataArray([p.area for p in props], dims=['label'], coords={'label': labelprops})  # Number of pixels of the region.
            if area.size == 0:
                raise ValueError(f'No objects were detected. Try changing radius or min_size_quartile parameters.')
            
            min_area = np.percentile(area, self.min_size_quartile*100)
            
            
            keep_labels = labelprops.where(area>=min_area, drop=True)
            keep_where = np.isin(labels_wrapped[:,:,:,each_depth], keep_labels)
            
            
            out_labels = xr.DataArray(np.where(keep_where==False, 0, labels_wrapped[:,:,:,each_depth])
                                      , dims=binary_images.isel(depth = each_depth).drop('depth').dims, 
                                        coords=binary_images.isel(depth = each_depth).drop('depth').coords)
    
            # Convert images to binary. All positive values == 1, otherwise == 0
            binary_labels.loc[dict(depth = depth_levels[each_depth])] = out_labels.where(out_labels==0, drop=False, other=1)

        return area, min_area, binary_labels, N_initial


    def _label_either(self, data, **kwargs):
        if isinstance(data, dsa.Array):
            try:
                from dask_image.ndmeasure import label as label_dask
                def label_func(a, **kwargs):
                    # print(a)
                    ids, num = label_dask(a, **kwargs)
                    return ids
            except ImportError:
                raise ImportError(
                    "Dask_image is required to use this function on Dask arrays. "
                    "Either install dask_image or else call .load() on your data."
                )
        else:

            label_func = label_np
            
            data_array = copy.deepcopy(data)
            
            empty_data = np.zeros((np.shape(data_array)))
            
            if len(np.shape(data)) == 4:
                
                
                
                def match_and_relabel(prev_labels, current_labels, max_label, t ):
                    
                    print(t)
                    
                    prev_labels = prev_labels.astype(np.int32)
                    current_labels = current_labels.astype(np.int32)
                    
                    new_labels = np.zeros_like(current_labels)
                    


                    
                    prev_regions = find_objects(prev_labels)
                    curr_regions = find_objects(current_labels)
                    
                    assigned_labels = set()
                    relabel_map = {}  # Maps from original label in tn-1 to new label in tn
                    
                    for i, curr_region in enumerate(curr_regions):
                        
                        curr_region_mask = (current_labels == (i+1))
                        
                        overlaps = []
                        for j, prev_region in enumerate(prev_regions):
                            prev_region_mask = (prev_labels == (j+1))
                            overlap = np.sum(curr_region_mask & prev_region_mask)
                            
                            if overlap > 0:
                                # print(overlap.data)
                                overlaps.append((overlap, j+1))
                        
                        if overlaps:
                            # print()
                            # **Check for Merging:**
                            # If multiple regions from `tn-1` overlap with this region in `tn`, 
                            # we consider it a merging case. 
                            if len(overlaps) > 1:
                                smallest_label = min([label for _, label in overlaps])
                                
                                # Assign the smallest label from tn-1 to the current region
                                new_labels[curr_region_mask] = smallest_label
                                
                                # Print out merging information
                                print(f"Region {smallest_label} in tn is formed by merging regions {[label for _, label in overlaps]} from tn-1.")
                
                                # Map all contributing previous labels to the smallest label
                                for _, prev_label in overlaps:
                                    relabel_map[prev_label] = smallest_label
                            
                            # **Check for Moving and Growing/Shrinking:**
                            # If there's a significant overlap with only one previous region,
                            # it's either a move or a grow/shrink (if the size has changed).
                            else:
                                main_overlap_label = overlaps[0][1]
                                
                                # Assign the same label to the matching region
                                new_labels[curr_region_mask] = main_overlap_label
                                
                                # If the region size has changed, it's growing/shrinking; otherwise, it's moving.
                                # This can be tracked by comparing the volume (number of voxels) of the regions.
                                prev_region_volume = np.sum(prev_labels == main_overlap_label)
                                curr_region_volume = np.sum(curr_region_mask)
                                if curr_region_volume > prev_region_volume:
                                    print(f"Region {main_overlap_label} is growing.")
                                elif curr_region_volume < prev_region_volume:
                                    print(f"Region {main_overlap_label} is shrinking.")
                                else:
                                    print(f"Region {main_overlap_label} is moving.")
                        
                        # **Check for New Regions (Splitting from Existing):**
                        # If there's no overlap with any previous region, this is a new region or a part of a split.
                        else:
                            max_label += 1
                            new_labels[curr_region_mask] = max_label
                            
                            # Print out splitting information
                            print(f"A new region {max_label} is forming anew.")
    
    
                    
                    # **Relabel the previous step's regions according to the merging rule:**
                    # This ensures that all contributing regions in tn-1 get the same label if they merge in tn.
                    for prev_label, new_label in relabel_map.items():
                        prev_labels[prev_labels == prev_label] = new_label
                    
                    return new_labels, prev_labels, max_label 
                
                
                def match_relabel_sina_vs(prev_labels, current_labels, max_label, t ):
                    prev_labels = prev_labels.astype(np.int32)
                    current_labels = current_labels.astype(np.int32)
                    
                    a,b,c = np.shape(prev_labels)
                    
                    for ai in range(a):
                        for bi in range(b):
                            for ci in range(c): 
                                if prev_labels[ai,bi,ci] != 0 and current_labels[ai,bi,ci] != 0: 
                                    if prev_labels[ai,bi,ci] != current_labels[ai,bi,ci]:
                                        
                                        
                                        label_0 = prev_labels[ai,bi,ci]
                                        label_1 = current_labels[ai,bi,ci]
                                        
                                        # Get the indices where t3d meets your condition (e.g., values greater than 0.5)
                                        indices = np.where(current_labels == label_1)
                                        current_labels[indices] = label_0
                                
                                        
                                        
                                    # label_0 = prev_labels[ai,bi,ci] 
                                    
                                    # indices = np.where(current_labels == label_0)
                                    
        
                                    # # Check where the corresponding values in t3d_2 are non-zero
                                    # non_zero_mask = prev_labels[indices] != 0
                                        
                                    # # Modify the values in t3d_2 at these indices to 3 where the non-zero mask is True
                                    # prev_labels[indices] = np.where(non_zero_mask, 3, prev_labels[indices])
                                    
                                    else:
                                       label_0 = max_label
                                else:
                                    label_0 = max_label
                                         
                    
                    return current_labels, prev_labels, label_0 
                
                
                def track_and_print_changes(prev_labels, current_labels):
                    # Ensure labels are integers
                    prev_labels = prev_labels.astype(np.int32)
                    current_labels = current_labels.astype(np.int32)
                    # print(np.unique(prev_labels),np.unique(current_labels))
                    # prev_regions = find_objects(prev_labels)
                    # curr_regions = find_objects(current_labels)
                    
                    l_prev = np.unique(prev_labels)
                    l_curr = np.unique(current_labels)
                    if np.array_equal(l_prev, l_curr) :
                        if len(l_prev) == 1 and l_prev[0] == 0:
                            print('No MHW region')
                        else: 
                            print(f'MHW region {l_prev[1:]} still here')
                    
                    else:
                        # Convert arrays to sets for comparison
                        set1 = set(l_prev)
                        set2 = set(l_curr)
                    
                        # Find new values in array2
                        new_values = set2 - set1
                        if new_values:
                            print(f"Region {list(new_values)[0]} arrives" )
                    
                        # Find values in array1 that are not in array2
                        gone_values = set1 - set2
                        if gone_values:
                            print(f"Region {list(gone_values)[0]} is gone")
        
                    
        
                    
                    return
            
                
                
                
                    
                                    
                    
                
                def propagate_labels(data):
                    # Assuming data is a 4D array with dimensions [time, lon, lat, depth]
                    time_series = data  # The 4D array
                    
                    # Initialize the relabeled time series array
                    relabelled_series = np.zeros_like(time_series)
                    
                    
                    
                    # Process the first time step
                    relabelled_series[0] = time_series[0]  # First time step stays as is
                    
                    ### starting label 
                    max_label =  np.max(relabelled_series[0])
                    
                    # Iterate over the time dimension
                    for t in range(1, time_series.shape[0]):
                        prev_labels = relabelled_series[t-1]
                        current_labels = time_series[t]
                        
                        new_labels, prev_labels, max_label  = match_and_relabel(prev_labels, current_labels, max_label, t )
                        # new_labels, prev_labels, max_label  = match_relabel_sina_vs(prev_labels, current_labels, max_label, t )
                        
                        track_and_print_changes(prev_labels, new_labels)
                        print('')
                        relabelled_series[t] = new_labels
                        relabelled_series[t-1] = prev_labels
                        
                    num = np.unique(relabelled_series) #[1:]
                    return relabelled_series,num
                    
        
                
              
                
              
                
                consistent_labels, num = propagate_labels(data_array) 
               
                return consistent_labels, num
            
        

            else:
                empty_data = label_func(data, **kwargs)
            
                return empty_data 


    def _wrap(self, labels):
        ''' Impose periodic boundary and wrap labels'''
        # print(labels)
        # print(np.shape(labels))
        
        first_column = labels[..., 0]
        last_column = labels[..., -1]
        
        # print(first_column)
        # print(last_column)

        unique_first = np.unique(first_column[first_column>0])

        # This loop iterates over the unique values in the first column, finds the location of those values in 
        # the first columnm and then uses that index to replace the values in the last column with the first column value
        for i in enumerate(unique_first):
            first = np.where(first_column == i[1])
            last = last_column[first[0], first[1]]
            bad_labels = np.unique(last[last>0])
            replace = np.isin(labels, bad_labels)
            labels[replace] = i[1]

        labels_wrapped = np.unique(labels, return_inverse=True)[1].reshape(labels.shape)

        # recalculate the total number of labels 
        N = np.max(labels_wrapped)

        return labels_wrapped, N
    

 








###############################
# =============================================================================
#    
# =============================================================================
    

class Tracker3D:

        
    def __init__(self, da, mask, radius, min_size_quartile, timedim, xdim = 'xh', ydim = 'yh', positive=True):
        
        self.da = da
        self.mask = mask
        self.radius = radius
        self.min_size_quartile = min_size_quartile
        self.timedim = timedim
        self.xdim = xdim
        self.ydim = ydim   
        self.positive = positive
        
        if ((timedim, ydim, xdim) != da.dims):
            try:
                da = da.transpose(timedim, ydim, xdim) 
            except:
                raise ValueError(f'Ocetrac currently only supports 3D DataArrays. The dimensions should only contain ({timedim}, {xdim}, and {ydim}). Found {list(da.dims)}')

            
    def track(self):
        '''
        Label and track image features.
        
        Parameters
        ----------
        da : xarray.DataArray
            The data to label.

        mask : xarray.DataArray
            The mask of ponts to ignore. Must be binary where 1 = true point and 0 = background to be ignored. 

        radius : int
            The size of the structuring element used in morphological opening and closing. Radius specified by the number of grid units.

        min_size_quartile : float
            The quantile used to define the threshold of the smallest area object retained in tracking. Value should be between 0 and 1.

        timedim : str
            The name of the time dimension
        
        xdim : str
            The name of the x dimension

        ydim : str
            The namne of the y dimension
            
        positive : bool
            True if da values are expected to be positive, false if they are negative. Default argument is True

        Returns
        -------
        labels : xarray.DataArray
            Integer labels of the connected regions.
        '''

        if (self.mask == 0).all():
            raise ValueError('Found only zeros in `mask` input. The mask should indicate valid regions with values of 1')

        # Convert data to binary, define structuring element, and perform morphological closing then opening
        binary_images = self._morphological_operations()

        # Apply mask
        binary_images_with_mask  = _apply_mask(binary_images,self.mask) # perhaps change to method? JB

        # Filter area
        area, min_area, binary_labels, N_initial = self._filter_area(binary_images_with_mask)

        # Label objects
        labels, num = self._label_either(binary_labels, return_num= True, connectivity=3)

        # Wrap labels
        grid_res = abs(self.da[self.xdim][1]-self.da[self.xdim][0])
        if self.da[self.xdim][-1]-self.da[self.xdim][0] >= 360-grid_res:
            labels_wrapped, N_final = self._wrap(labels)
        else:
            labels_wrapped = labels
            N_final = np.max(labels)
                
        # Final labels to DataArray
        new_labels = xr.DataArray(labels_wrapped, dims=self.da.dims, coords=self.da.coords)   
        new_labels = new_labels.where(new_labels!=0, drop=False, other=np.nan)

        ## Metadata

        # Calculate Percent of total object area retained after size filtering
        sum_tot_area = int(np.sum(area.values))

        reject_area = area.where(area<=min_area, drop=True)
        sum_reject_area = int(np.sum(reject_area.values))
        percent_area_reject = (sum_reject_area/sum_tot_area)

        accept_area = area.where(area>min_area, drop=True)
        sum_accept_area = int(np.sum(accept_area.values))
        percent_area_accept = (sum_accept_area/sum_tot_area)

        new_labels = new_labels.rename('labels')
        new_labels.attrs['inital objects identified'] = int(N_initial)
        new_labels.attrs['final objects tracked'] = int(N_final)
        new_labels.attrs['radius'] = self.radius
        new_labels.attrs['size quantile threshold'] = self.min_size_quartile
        new_labels.attrs['min area'] = min_area
        new_labels.attrs['percent area reject'] = percent_area_reject
        new_labels.attrs['percent area accept'] = percent_area_accept

        print('inital objects identified \t', int(N_initial))
        print('final objects tracked \t', int(N_final))

        return new_labels

    def collect_surface_stats(self, labels):
        """Collects surface statistics"""
    
        ids = np.unique(labels)
        ids = np.array([id for id in ids if ~np.isnan(id)])
        
        dataframes = []
        num_events = len(ids)
    
        for i in range(num_events):
            event = labels.where(labels == ids[i], drop=True)
            df = self._get_surface_stats(event) #self.da or sst
            dataframes.append(df)
            #break
    
        dff = pd.concat(dataframes, ignore_index=True)
        return dff

    ### PRIVATE METHODS - not meant to be called by user ###

    def _compute_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
        """Compute the x and y resolution in km for a geographic degree resolution
    
        :param clat: the point latitude at which to compute the resolution
        :param clon: the point longitude at which to compute the resolution
        :param res: the geographic degree resolution
        :return: distance in km between two points
        """
        
        dist = haversine_distances(
            np.deg2rad(np.array([(lat1, lon1)])),
            np.deg2rad(np.array([(lat2, lon2)])),
        )
        
        dist = dist * 6371000 / 1000  # multiply by Earth radius to get kilometers
        dist
        return dist[0][0]

    def _get_surface_stats(self, event):
        """ get surface stats for one event """
        
        # Initialize dictionary 
        mhw = {}
        mhw['id'] = [] # event label
        mhw['dates'] = [] # datetime format
        mhw['coords'] = [] # (lat, lon)
        mhw['centroid'] = []  # (lat, lon)
        mhw['duration'] = [] # [months]
        mhw['intensity_max'] = [] # [deg C]
        mhw['intensity_mean'] = [] # [deg C]
        mhw['intensity_min'] = [] # [deg C]
        mhw['intensity_cumulative'] = [] # [deg C]
        mhw['total_area'] = [] # [km2]
        mhw['distance'] = [] 
        
        # TO ADD:
        # mhw['rate_onset'] = [] # [deg C / month]
        # mhw['rate_decline'] = [] # [deg C / month]
        
        mhw["id"].append(int(np.nanmedian(event.values)))
        mhw["dates"].append(event.time.values)
        mhw["duration"] = event.time.shape[0]
        mhw["total_area"] = int(event.sum("time").count((self.ydim,self.xdim)).values)
        
        centroid_list = []
        distance_list = []
        
        prev_coords = None  # Initialize previous coordinates as None
        
        for time in event.time.values:
            mhw_slice = event.sel(time=time)
            
            # Create latitude and longitude grids
            lon_grid, lat_grid = np.meshgrid(np.arange(0, mhw_slice[self.xdim].shape[0]), np.arange(0, mhw_slice[self.ydim].shape[0]))
            lon_flat = lon_grid.flatten()
            lat_flat = lat_grid.flatten()
            data_flat = mhw_slice.values.flatten()
        
            # Compute the centroid
            centroid_lon = np.nansum(lon_flat * data_flat) / np.nansum(data_flat)
            centroid_lat = np.nansum(lat_flat * data_flat) / np.nansum(data_flat)

            coords = [int(mhw_slice[self.xdim][round(centroid_lon)].values), int(mhw_slice[self.ydim][round(centroid_lat)].values)]
            centroid_list.append(coords)
            
            # Extract lat and lon from the current coordinates
            lon2, lat2 = coords  # lon2 is the first element, lat2 is the second
            
            # Calculate distance from previous coordinates, if available
            if prev_coords is not None:
                lon1, lat1 = prev_coords  
                distance = self._compute_distance(lat1, lon1, lat2, lon2)
                distance_list.append(distance)
            
            prev_coords = coords  
        
        # Add a None or NaN for the first distance since it's undefined
        distance_list.insert(0, np.nan)
        #distance_list.insert(0, None)  # or use `np.nan` if you prefer
        
        mhw['centroid'].append(centroid_list)
        mhw['distance'].append(distance_list)
        
        # Process intensity metrics using try-except to handle ValueError
        event_ssta = self.da.where(event > 0, drop=True)
        
        try:
            mhw['intensity_mean'].append(event_ssta.mean((self.ydim, self.xdim)).values)
        except ValueError:
            mhw['intensity_mean'].append(np.nan)
        try:
            mhw['intensity_max'].append(event_ssta.max((self.ydim, self.xdim)).values)
        except ValueError:
            mhw['intensity_max'].append(np.nan)
        try:
            mhw['intensity_min'].append(event_ssta.min((self.ydim, self.xdim)).values)
        except ValueError:
            mhw['intensity_min'].append(np.nan)
        try:
            # print(np.nansum(event_ssta))
            mhw['intensity_cumulative'].append(np.nansum(event_ssta))
        except ValueError:
            mhw['intensity_cumulative'].append(np.nan)
            
        coords = event.stack(z=(self.ydim, self.xdim))
        coord_pairs = [(coords.isel(time=t[0]).dropna(dim='z', how='any').z[self.ydim].values, 
                          coords.isel(time=t[0]).dropna(dim='z', how='any').z[self.xdim].values) for t in enumerate(event.time)]
        
        mhw['coords'].append(coord_pairs)
        
        mhw = pd.DataFrame(dict([(name, pd.Series(data)) for name,data in mhw.items()]))
        return mhw

    def _morphological_operations(self): 
        '''Converts xarray.DataArray to binary, defines structuring element, and performs morphological closing then opening.
        Parameters
        ----------
        da     : xarray.DataArray
                The data to label
        radius : int
                Length of grid spacing to define the radius of the structing element used in morphological closing and opening.

        '''

        # Convert images to binary. All positive values == 1, otherwise == 0
        if self.positive == True:
            bitmap_binary = self.da.where(self.da>0, drop=False, other=0)
        
        elif self.positive == False:
            bitmap_binary = self.da.where(self.da<0, drop=False, other=0)
    
        bitmap_binary = bitmap_binary.where(bitmap_binary==0, drop=False, other=1)

        # Define structuring element
        diameter = self.radius*2
        x = np.arange(-self.radius, self.radius+1)
        x, y = np.meshgrid(x, x)
        r = x**2+y**2 
        se = r<self.radius**2

        def binary_open_close(bitmap_binary):
            bitmap_binary_padded = np.pad(bitmap_binary,
                                          ((diameter, diameter), (diameter, diameter)),
                                          mode='wrap')
            s1 = scipy.ndimage.binary_closing(bitmap_binary_padded, se, iterations=1)
            s2 = scipy.ndimage.binary_opening(s1, se, iterations=1)
            unpadded= s2[diameter:-diameter, diameter:-diameter]
            return unpadded

        mo_binary = xr.apply_ufunc(binary_open_close, bitmap_binary,
                                   input_core_dims=[[self.ydim, self.xdim]],
                                   output_core_dims=[[self.ydim, self.xdim]],
                                   output_dtypes=[bitmap_binary.dtype],
                                   vectorize=True,
                                   dask='parallelized')
        return mo_binary


    def _filter_area(self, binary_images):
        '''calculatre area with regionprops'''

        def get_labels(binary_images):
            blobs_labels = self._label_either(binary_images, background=0)
            return blobs_labels

        labels = xr.apply_ufunc(get_labels, binary_images,
                                input_core_dims=[[self.ydim, self.xdim]],
                                output_core_dims=[[self.ydim, self.xdim]],
                                output_dtypes=[binary_images.dtype],
                                vectorize=True,
                                dask='parallelized')


        labels = xr.DataArray(labels, dims=binary_images.dims, coords=binary_images.coords)
        labels = labels.where(labels>0, drop=False, other=np.nan)  

        # The labels are repeated each time step, therefore we relabel them to be consecutive
        for i in range(1, labels.shape[0]):
            labels[i,:,:] = labels[i,:,:].values + labels[i-1,:,:].max().values

        labels = labels.where(labels>0, drop=False, other=0)  
        labels_wrapped, N_initial = self._wrap(np.array(labels))

        # Calculate Area of each object and keep objects larger than threshold
        props = regionprops(labels_wrapped.astype('int'))
        
        labelprops = [p.label for p in props]
        labelprops = xr.DataArray(labelprops, dims=['label'], coords={'label': labelprops}) 
        
        area = xr.DataArray([p.area for p in props], dims=['label'], coords={'label': labelprops})  # Number of pixels of the region.

        if area.size == 0:
            raise ValueError(f'No objects were detected. Try changing radius or min_size_quartile parameters.')
        
        min_area = np.percentile(area, self.min_size_quartile*100)
        print(f'minimum area: {min_area}') 
        
        keep_labels = labelprops.where(area>=min_area, drop=True)
        keep_where = np.isin(labels_wrapped, keep_labels)
        out_labels = xr.DataArray(np.where(keep_where==False, 0, labels_wrapped), dims=binary_images.dims, coords=binary_images.coords)

        # Convert images to binary. All positive values == 1, otherwise == 0
        binary_labels = out_labels.where(out_labels==0, drop=False, other=1)

        return area, min_area, binary_labels, N_initial


    def _label_either(self, data, **kwargs):
        if isinstance(data, dsa.Array):
            try:
                from dask_image.ndmeasure import label as label_dask
                def label_func(a, **kwargs):
                    ids, num = label_dask(a, **kwargs)
                    return ids
            except ImportError:
                raise ImportError(
                    "Dask_image is required to use this function on Dask arrays. "
                    "Either install dask_image or else call .load() on your data."
                )
        else:
            label_func = label_np
        return label_func(data, **kwargs)


    def _wrap(self, labels):
        ''' Impose periodic boundary and wrap labels'''
        first_column = labels[..., 0]
        last_column = labels[..., -1]

        unique_first = np.unique(first_column[first_column>0])

        # This loop iterates over the unique values in the first column, finds the location of those values in 
        # the first columnm and then uses that index to replace the values in the last column with the first column value
        for i in enumerate(unique_first):
            first = np.where(first_column == i[1])
            last = last_column[first[0], first[1]]
            bad_labels = np.unique(last[last>0])
            replace = np.isin(labels, bad_labels)
            labels[replace] = i[1]

        labels_wrapped = np.unique(labels, return_inverse=True)[1].reshape(labels.shape)

        # recalculate the total number of labels 
        N = np.max(labels_wrapped)

        return labels_wrapped, N