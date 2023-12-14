import numpy as np
import xarray as xr
from datetime import datetime
import pylab as py
import gc
from scipy.interpolate import interp1d
import sys
import os
import time as ti
import gridfill
sys.path.append('./module/')
import logruns as logruns



def gridfill_each_level(lat_lon_field, itermax=1000, verbose=False):
    #### copied from Clare's MDTF function
    """
    Fill missing values in lat-lon grids with values derived by solving Poisson's equation
    using a relaxation scheme.

    Args:
        lat_lon_field(np.ndarray): 2D array to apply gridfill on
        itermax(int): maximum iteration for poisson solver
        verbose(bool): verbose level of poisson solver

    Returns:
        A 2D array of the same dimension with all nan filled.
    """
    if np.isnan(lat_lon_field).sum() == 0:
        return lat_lon_field

    lat_lon_filled, converged = gridfill.fill(
        grids=np.ma.masked_invalid(lat_lon_field), xdim=1, ydim=0, eps=0.001,
        cyclic=True, itermax=itermax, verbose=verbose)

    return lat_lon_filled


def gridfill_all_levels_each_time(new_variable_each_time, logging_object):
    
    PLEVS, LAT, LON = new_variable_each_time.shape
    
    start0 = ti.time()
    for plev in range(PLEVS):
        
        lat_lon_field = new_variable_each_time[plev, ...]
        new_variable_each_time[plev, ...] = gridfill_each_level(lat_lon_field, itermax=30000, verbose=False)   
    end0       = ti.time() 
    time_taken = (end0-start0)  
                              
    return new_variable_each_time                      
    
                         

def vertical_interpolation_to_pressure(ds, variable, save_topography=False, logging_object=None, \
                                       new_lev = np.array([200, 300]), var='U'):
    
    start_interp= ti.time()
    
    KIND='linear'
    ai, bi, PS, P0 = ds.hyai.values, ds.hybi.values, ds.PS.values, ds.P0.values
    
    TIME, OLD_LEV, LAT, LON = variable.shape
    NEW_LEVS       = len(new_lev)
    new_variable   = np.zeros((TIME, NEW_LEVS, LAT, LON))

                         
    if save_topography:
        topography = np.copy(new_variable)
    else:
        topography = None
                         
    for t in range(TIME):
        start0 = ti.time()
        for la in range(LAT):
            for lo in range(LON):
                         
                P             = (ai*(P0) + bi*(PS[t,la,lo]))
                P             = (P[1:]+P[:-1])/2   
                old_lev       =  P/100 ### converting from Pa to hPa
                f_interp      =  interp1d(old_lev, variable[t,:,la, lo], fill_value=np.nan, bounds_error=False, kind=KIND)
                new_variable[t, :, la, lo] = f_interp(new_lev)
            
        end0       = ti.time() 
        time_taken = (end0-start0)
        logging_object.write('%d of %d interpolated - computed in %1.1f sec'%(t, TIME, time_taken))
             
        ##### Replace the nan values (i.e. boundary below topography) with smooth values from x-y boundary                
        if save_topography:
            ## save a copy for topography file where there no values for certain pressure cooridinates. 
            topography[t, ...] = new_variable[t, ...]            
        
        new_variable[t, ...]   = gridfill_all_levels_each_time(new_variable[t, ...], logging_object)
                         
                         
    if save_topography:
        topography[~np.isnan(topography)]  = 1
        ### This is a time varying 4d array (time, pressure, lat, lon) but it should be more or less constant. Can save a time mean value as well. 
        ### Its 1 above topography and nan below topography. 
        ### To-do : Use this variable later for doing topography correction during computation of vertically avergaed LWA budget later.
                         
    for var in [v for v in list(locals()) if v not in ['new_variable', 'topography', 'new_lev']]:
        del locals()[var]
    gc.collect()
    
    end_interp= ti.time()
    time_diff = (end_interp - start_interp)/60
    logging_object.write('%s interpolated in %d min'%(var, time_diff))
    logging_object.write('---------------------------------------')
        
    return new_variable, topography, new_lev


def similar_xarray_dataset(ds, new_lev, new_lev_name='plev'):
    ds_copy = xr.Dataset()
    for dim, size in ds.sizes.items():       # Copy dimensions
        ds_copy[dim] = np.arange(size)
    for coord, values in ds.coords.items():  # Copy coordinates
        ds_copy[coord] = values
    ds_copy.attrs = ds.attrs.copy()          # Copy attributes
    ds_copy = ds_copy.drop_dims('lev').drop_dims('ilev').drop_dims('nbnd') \
    ## looks like if you remove the dimension sufficiently removes the indexes and coordinates as well.
    ds_copy = ds_copy.assign_coords({new_lev_name: new_lev})         \
    ## looks like if you add the dimension it adds the indexes and coordinates as well. 
    
    new_coordinate_system = ['time', new_lev_name, 'lat', 'lon']
    for var in new_coordinate_system:
        
        values = new_lev if var==new_lev_name else ds[var].values
        attrs  = {'units': 'hPa', 'direction': 'downwards'} if var==new_lev_name else ds[var].attrs
        ds_copy[var]   = xr.DataArray(values, \
                            dims   = (var), \
                            coords = {key:ds_copy.coords[key].values for key in [var]},\
                            attrs  = attrs) 
    ds_copy['PS']      = ds['PS'] ### Saving the surface pressure
    return ds_copy, new_coordinate_system


def update_new_variable_to_pressure(var, new_data, ds, ds_copy, default_attrs=None, new_coordinate_system=['time', 'plev', 'lat', 'lon']): ## new_data={var: data_array}
    
    
    if default_attrs is None:
        default_attrs = ds[var].attrs
        
    ds_copy[var]   = xr.DataArray(new_data, \
            dims   = tuple(new_coordinate_system), \
            coords = {key:ds_copy.coords[key].values for key in new_coordinate_system}, \
            attrs  = default_attrs.update({'description': '%s interpolated from model levels to pressure coords in hPa'%(var)}))   
    
    del new_data
    gc.collect()
    
    return ds_copy


def show_vars(file = "NHISTfrc2_f09_tn14_20191025.cam.h3.1950-01-01-21600.nc"):
    source = './NorESM_data_sample/'
    dataset = xr.open_dataset(source+file)
    return dataset

mapval ={'U': True, 'V':False, 'T':False}


if __name__ == "__main__": 
    
    start = ti.time()
    
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")    
    
    logfilename    = formatted_datetime+'-'+os.path.basename(__file__).split('/')[-1].split('.py')[0]
    logging_object = logruns.default_log(logfilename = logfilename,  log_directory = './logs/')

    
    file       = "NHISTfrc2_f09_tn14_20191025.cam.h3.1950-01-01-21600.nc"
    ds         = show_vars(file = file)
    
    varsi            = [sys.argv[1]] ## ['V', 'U', 'T', 'Z3']
    save_topographys = [mapval[var] for var in varsi]  ## [True, False, False, False]
    
    new_lev    = np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, \
                            750, 725, 700, 650, 600, 550, 500, 450, 400, 350, \
                            300, 250, 225, 200, 175, 150, 125, 100, 70, 50, \
                            30,   20,  10,   7,   5,   3,])  ##### ERA5 pressure levels 
    ### --> Not included 1 and 2 hPa pressure levels because stratosphere is not so well resolved in the model.
    
    new_lev = new_lev[::-1] ## downwards
    flag    = -1
    
    ds_copy, new_coordinate_system = similar_xarray_dataset(ds, new_lev, 'plev')
    
    for var, save_topography in zip(varsi, save_topographys):
                
            logging_object.write('########################################')
            logging_object.write('---> %s <------'%(var))


            interpolated_val, topography, new_lev = vertical_interpolation_to_pressure(\
                                                    ds=ds, variable = ds[var].values, new_lev=new_lev, \
                                                    save_topography=save_topography, logging_object=logging_object, var=var) 


            ds_copy = update_new_variable_to_pressure(var, interpolated_val, ds, ds_copy, default_attrs=None, \
                                                      new_coordinate_system = new_coordinate_system)

            if topography is not None:
                ds_copy = update_new_variable_to_pressure('topo', topography, ds, ds_copy, default_attrs={}, new_coordinate_system = new_coordinate_system) 

            
            ds.close()
            ds_copy.to_netcdf(path='./%s_poisson_filled_%s'%(var))
            ds_copy.close()

            gc.collect()
    
    end = ti.time()
    total_time_taken = (end-start)/(60)
    logging_object.write('Congratulations - You had a good day today !! %1.1f min'%(total_time_taken))
    
    
