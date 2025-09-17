#==============================================================================
# By Dr. Sandro W. Lubis, May 2024
# Computes the Local Wave Activity (LWA) budget as presented in the submitted manuscript:
# Lubis, S. W., et al. 2025, "Cloud-Radiative Effects Significantly Increase Wintertime
# Atmospheric Blocking in the Euro-Atlantic Sector" (in review, Nature Communications)
# Earth Scientist, PNNL
# Input: daily-mean data
# Variables: zonal wind (U, m/s), meridional wind (V, m/s), air temperature (T, K)
#==============================================================================

import numpy as np
import xarray as xr
from numpy import dtype
from math import pi
from netCDF4 import Dataset
import datetime as dt
from falwa.oopinterface import QGFieldNHN22
import falwa.utilities as utilities
import datetime as dt

year = range(1980, 2022)

for i in year:
    
    # --- Load the zonal wind and QGPV at 240hPa --- #
    u_file = Dataset('../U/intp/U.' + str(i) + "_plev.nc", mode='r')
    v_file = Dataset('../V/intp/V.' + str(i) + "_plev.nc", mode='r')
    t_file = Dataset('../T/intp/T.' + str(i) + "_plev.nc", mode='r')
    
    time_array = u_file.variables['time'][:]
    time_units = u_file.variables['time'].units
    time_calendar = u_file.variables['time'].calendar
    ntimes = time_array.shape[0]
    
    print('Dimension of time: {}'.format(time_array.size))
    
    xlon = u_file.variables['lon'][:]
    
    # latitude has to be in ascending order
    ylat = u_file.variables['lat'][:]
    if np.diff(ylat)[0]<0:
        print('Flip ylat.')
        ylat = ylat[::-1]
    
    # pressure level has to be in descending order (ascending height)
    plev = u_file.variables['level'][:] #/100.
    if np.diff(plev)[0]>0:
        print('Flip plev.')    
        plev = plev[::-1]
    
    nlon = xlon.size
    nlat = ylat.size
    nlev = plev.size
    
    clat = np.cos(np.deg2rad(ylat))     # cosine latitude
    p0 = 1000.                          # surface pressure [hPa]
    kmax = 49                           # number of grid points for vertical extrapolation (dimension of height)
    dz = 1000.                          # differential height element
    height = np.arange(0,kmax)*dz       # pseudoheight [m]
    dphi = np.diff(ylat)[0]*pi/180.     # differential latitudinal element
    dlambda = np.diff(xlon)[0]*pi/180.  # differential latitudinal element
    hh = 7000.                          # scale height
    cp = 1004.                          # heat capacity of dry air
    rr = 287.                           # gas constant
    omega = 7.29e-5                     # rotation rate of the earth
    aa = 6.378e+6                       # earth radius
    prefactor = np.array([np.exp(-z/hh) for z in height[1:]]).sum() # integrated sum of density from the level
                                                                    #just above the ground (z=1km) to aloft
    npart = nlat                        # number of partitions to construct the equivalent latitude grids
    maxits = 100000                     # maximum number of iteration in the SOR solver to solve for reference state
    tol = 1.e-5                         # tolerance that define convergence of solution
    rjac = 0.95                         # spectral radius of the Jacobi iteration in the SOR solver.              
    jd = nlat//2+1                      # (one plus) index of latitude grid point with value 0 deg
    eq_boundary_index = 5
                                        # This is to be input to fortran code. The index convention is different.
    
    # === Outputing files ===
    output_fname = 'output.' + str(i) + '.nc'
    output_file = Dataset(output_fname, 'w')
    output_file.createDimension('levelist',kmax)
    output_file.createDimension('latitude',nlat)
    output_file.createDimension('longitude',nlon)
    output_file.createDimension('time',ntimes) 
    plevs = output_file.createVariable('levelist',dtype('float32').char,('levelist',)) # Define the coordinate variables
    lats = output_file.createVariable('latitude',dtype('float32').char,('latitude',)) # Define the coordinate variables
    lons = output_file.createVariable('longitude',dtype('float32').char,('longitude',))
    times = output_file.createVariable('time',dtype('int').char,('time',))    
    plevs.units = 'hPa' 
    lats.units = 'degrees_north'
    lons.units = 'degrees_east'
    times.units = time_units
    times.calendar = time_calendar
    plevs[:] = p0 * np.exp(-height/hh)
    lats[:]  = ylat 
    lons[:]  = xlon
    times[:] = time_array
    qgpv = output_file.createVariable('qgpv',dtype('float32').char,('time','levelist','latitude','longitude'))
    qgpv.units = '1/s' 
    interpolated_u = output_file.createVariable('interpolated_u',dtype('float32').char,('time','levelist','latitude','longitude'))
    interpolated_u.units = 'm/s' 
    interpolated_v = output_file.createVariable('interpolated_v',dtype('float32').char,('time','levelist','latitude','longitude'))
    interpolated_v.units = 'm/s'
    interpolated_theta = output_file.createVariable('interpolated_theta',dtype('float32').char,('time','levelist','latitude','longitude'))
    interpolated_theta.units = 'K' 
    qref = output_file.createVariable('qref',dtype('float32').char,('time','levelist','latitude'))
    qref.units = '1/s' 
    uref = output_file.createVariable('uref',dtype('float32').char,('time','levelist','latitude'))
    uref.units = 'm/s' 
    ptref = output_file.createVariable('ptref',dtype('float32').char,('time','levelist','latitude'))
    ptref.units = 'K' 
    lwa = output_file.createVariable('lwa',dtype('float32').char,('time','levelist','latitude','longitude'))
    lwa.units = 'm/s'
    adv_flux_f1 = output_file.createVariable('adv_flux_f1',dtype('float32').char,('time','latitude','longitude'))
    adv_flux_f1.units = 'm**2/s**2'    
    adv_flux_f2 = output_file.createVariable('adv_flux_f2',dtype('float32').char,('time','latitude','longitude'))
    adv_flux_f2.units = 'm**2/s**2'  
    adv_flux_f3 = output_file.createVariable('adv_flux_f3',dtype('float32').char,('time','latitude','longitude'))
    adv_flux_f3.units = 'm**2/s**2'  
    adv_flux_conv = output_file.createVariable('adv_flux_conv',dtype('float32').char,('time','latitude','longitude'))
    adv_flux_conv.units = 'm/s**2'  
    divergence_eddy_momentum_flux = output_file.createVariable('divergence_eddy_momentum_flux',dtype('float32').char,('time','latitude','longitude'))
    divergence_eddy_momentum_flux.units = 'm/s**2'  
    meridional_heat_flux = output_file.createVariable('meridional_heat_flux',dtype('float32').char,('time','latitude','longitude'))
    meridional_heat_flux.units = 'm/s**2' 
    lwa_baro = output_file.createVariable('lwa_baro',dtype('float32').char,('time','latitude','longitude'))
    lwa_baro.units = 'm/s' 
    u_baro = output_file.createVariable('u_baro',dtype('float32').char,('time','latitude','longitude'))
    u_baro.units = 'm/s'
    
    #itstamp = [dt.datetime(i,1,1,0,0) + dt.timedelta(seconds=6*3600) * tt for tt in range(ntimes)]
    #plev_selected = 10 # selected pressure level to display
    #tstep_selected = 0
    
    for tstep in range(ntimes):  # or ntimes
        
        uu = u_file.variables['U'][tstep, :, :, :].data
        vv = v_file.variables['V'][tstep, :, :, :].data
        tt = t_file.variables['T'][tstep, :, :, :].data
    
        qgfield_object = QGFieldNHN22(xlon, ylat, plev, uu, vv, tt, eq_boundary_index=eq_boundary_index, northern_hemisphere_results_only=False)
        equator_idx = qgfield_object.equator_idx
    
        qgpv[tstep, :, :, :], interpolated_u[tstep, :, :, :], interpolated_v[tstep, :, :, :], interpolated_theta[tstep, :, :, :], static_stability = qgfield_object.interpolate_fields(return_named_tuple=True)
    
        qref[tstep, :, :], uref[tstep, :, :], ptref[tstep, :, :] = qgfield_object.compute_reference_states(return_named_tuple=True)
    
        adv_flux_f1[tstep, :, :], \
        adv_flux_f2[tstep, :, :], \
        adv_flux_f3[tstep, :, :], \
        adv_flux_conv[tstep, :, :], \
        divergence_eddy_momentum_flux[tstep, :, :], \
        meridional_heat_flux[tstep, :, :], \
        lwa_baro[tstep, :, :], \
        u_baro[tstep, :, :], \
        lwa[tstep, :, :, :] \
            = qgfield_object.compute_lwa_and_barotropic_fluxes(return_named_tuple=True)
    
    output_file.close()
    print('Output {} timesteps of data to the file {}'.format(tstep + 1, output_fname))
