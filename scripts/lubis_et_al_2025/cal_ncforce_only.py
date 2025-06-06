"""
------------------------------------------
File name: cal_ncforce_only.py
Author: Clare Huang, Sandro Lubis
Description: This is the script responsible for the direct computation of diabatic contribution to LWA budget in
    Lubis et al. (2025) "Cloud-Radiative Effects Significantly Increase Wintertime Atmospheric Blocking in the
    Euro-Atlantic Sector" Eq.12 in Session 5 (Method). The diabatic heating rate \dot{Q} directly output from
    the model is first interpolated onto regular pseudo-height grid (i.e. the analysis grid for LWA). The integrand
    in Eq.12 (excluding the cosine factor cos(phi+phi')) is then precomputed and saved into the file
    `../../../Q_lwcre/Q`, which is the input to the method (new in version 2.1.0)
        >> QGField.compute_lwa_and_barotropic_fluxes(return_named_tuple=False, ncforce=q)
    The barotropic component of integrated diabatic term can be retrieved by QGField.ncforce_baro.
"""
import numpy as np
from numpy import dtype
from math import pi
from netCDF4 import Dataset
from falwa.oopinterface import QGFieldNHN22

year = range(1921, 1922)

for i in year:
    
    
    u_file = Dataset('../../../u/intp/regrid/U.' + str(i) + "_plev.nc", mode='r')
    v_file = Dataset('../../../v/intp/regrid/V.' + str(i) + "_plev.nc", mode='r')
    t_file = Dataset('../../../t/intp/regrid/T.' + str(i) + "_plev.nc", mode='r')
    q_file = Dataset('../../../Q_lwcre/Q.' + str(i) + ".plev.nc", mode='r')   #interpolated z
    
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
    plev = u_file.variables['level'][:]
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
 
    ncforce_baro = output_file.createVariable('ncforce_baro',dtype('float32').char,('time','latitude','longitude'))
    ncforce_baro.units = 'm/s2'
    

    
    for tstep in range(ntimes):  # or ntimes
        
        uu = u_file.variables['U'][tstep, :, :, :].data
        vv = v_file.variables['V'][tstep, :, :, :].data
        tt = t_file.variables['T'][tstep, :, :, :].data
        q = q_file.variables['Q'][tstep, :, :, :].data
    
        qgfield_object = QGFieldNHN22(xlon, ylat, plev, uu, vv, tt, eq_boundary_index=eq_boundary_index, northern_hemisphere_results_only=False)
        equator_idx = qgfield_object.equator_idx
    
        qgfield_object.interpolate_fields(return_named_tuple=False)
    
        qgfield_object.compute_reference_states(return_named_tuple=False)
        
        qgfield_object.compute_lwa_and_barotropic_fluxes(ncforce=q)
    
        qgfield_object.compute_lwa_and_barotropic_fluxes(return_named_tuple=False, ncforce=q)
            
        ncforce_baro[tstep, :, :] = qgfield_object.ncforce_baro    
    
    output_file.close()
    print('Output {} timesteps of data to the file {}'.format(tstep + 1, output_fname))
