# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:19:43 2024

@author: 24048369
"""
import copy
import xarray as xr
import numpy as np
import scipy.ndimage
from skimage.measure import regionprops 
from skimage.measure import label as label_np
import dask.array as dsa

def _apply_mask(binary_images, mask):
    binary_images_with_mask = binary_images.where(mask==1, drop=False, other=0)
    return binary_images_with_mask

class Tracker3D:
        
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
                

        # Final labels to DataArray
        # print(labels_wrapped)
        print(np.shape(labels_wrapped))
        
        labels_wrapped = np.transpose(labels_wrapped, (0, 3, 1, 2))
        print(np.shape(labels_wrapped))
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
        # print(np.shape(bitmap_binary))
        
        # Define structuring element
        diameter = self.radius*2
        x = np.arange(-self.radius, self.radius+1)
        x, y, z = np.meshgrid(x, x, x)
        
        r = x**2 + y**2 + z**2
        se = r<self.radius**2
        # print(np.shape(se))
        def binary_open_close(bitmap_binary):
            bitmap_binary_padded = np.pad(bitmap_binary,diameter,
                                          # ((diameter, diameter), 
                                          #  (diameter, diameter),
                                          #  (diameter, diameter)),
                                          mode='wrap')
            
            # print(np.shape(bitmap_binary_padded))
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
        # print(np.shape(labels))
        # The labels are repeated each time step, therefore we relabel them to be consecutive
        for i in range(1, labels.shape[0]):
            labels[i,:,:] = labels[i,:,:].values + labels[i-1,:,:].max().values

        labels = labels.where(labels>0, drop=False, other=0)  
        labels_wrapped, N_initial = self._wrap(np.array(labels))
        
        
        # print('other labesl warpped TEST ')
        # print(np.shape(labels_wrapped))
        # print('test over')
        # Calculate Area of each object and keep objects larger than threshold       
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
        # print('before loop - just surface')
        # print(binary_labels)
        
        # print('yes - surface area')
        
        # if area.size == 0:
        #     raise ValueError(f'No objects were detected. Try changing radius or min_size_quartile parameters.')
        
        # min_area = np.percentile(area, self.min_size_quartile*100)
        
        binary_labels = copy.deepcopy(binary_images)
        depth_levels = binary_labels.depth.data
        
        for each_depth in range( np.shape(labels_wrapped)[3]) :
            
            
            
            props = regionprops(labels_wrapped[:,:,:,each_depth].astype('int'))
            labelprops = [p.label for p in props]
            labelprops = xr.DataArray(labelprops, dims=['label'], coords={'label': labelprops}) 
            # print(labelprops)
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
            
            
        # print('after loop')
        # print(binary_labels)
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
            # print('else')
            # print(label_np)
            label_func = label_np
            
            # print(data)
            data_array = copy.deepcopy(data)
            
            empty_data = np.zeros((np.shape(data_array)))
            # print(np.shape(data))
            if len(np.shape(data)) == 4:
                
                
                for d in range(np.shape(data)[-1]):
                    empty_data[:,:,:,d],num = label_func(data_array[:,:,:,d], **kwargs)
                    
                for l in range(np.shape(data)[-2]):
                    empty_data[:,:,l,:],num = label_func(data_array[:,:,l,:], **kwargs)
                
                for l in range(np.shape(data)[-3]):
                    empty_data[:,l,:,:],num = label_func(data_array[:,l,:,:], **kwargs)
                    
                for d in range(np.shape(data)[-1]):
                    empty_data[:,:,:,d],num = label_func(data_array[:,:,:,d], **kwargs)
                    
                for t in range(np.shape(data)[0]):
                    empty_data[t,:,:,:],num = label_func(data_array[t,:,:,:], **kwargs)
                    
                return empty_data, num
                    

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
