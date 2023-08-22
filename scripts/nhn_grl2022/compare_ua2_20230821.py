"""
-------------------------------------------------------------------------------------------------------------------
File name: compare_ua2_20230821.py
Author: Clare Huang
Created on: 2023/8/21
Description: Sample script to reproduce plots in Neal et al (2022, GRL)
-------------------------------------------------------------------------------------------------------------------
"""
import os
import sys
import numpy as np
from math import pi
from netCDF4 import Dataset
from hn2016_falwa.oopinterface import QGFieldNHN22

sys.path.insert(0, os.getcwd())
data_dir = "../../github_data_storage/"
to_generate_data = True  # Set this to False if wave activity and fluxes are computed and stored in netCDF file already

# --- Load the zonal wind and QGPV at 240hPa --- #
u_file = Dataset(data_dir + '2022_01_u.nc', mode='r')
v_file = Dataset(data_dir + '2022_01_v.nc', mode='r')
t_file = Dataset(data_dir + '2022_01_t.nc', mode='r')
time_array = u_file.variables['time'][:]
time_units = u_file.variables['time'].units
time_calendar = u_file.variables['time'].calendar
ntimes = time_array.shape[0]

print('Dimension of time: {}'.format(time_array.size))

# --- Longitude, latitude and pressure grid ---
xlon = u_file.variables['longitude'][:]

# latitude has to be in ascending order
ylat = u_file.variables['latitude'][:]
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

# --- Coordinates ---
clat = np.cos(np.deg2rad(ylat))     # cosine latitude
p0 = 1000.                          # surface pressure [hPa]
kmax = 97                           # number of grid points for vertical extrapolation (dimension of height)
dz = 500.                           # differential height element
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
nd = nlat//2+1                      # (one plus) index of latitude grid point with value 0 deg
                                    # This is to be input to fortran code. The index convention is different.
eq_boundary_index = 5               # This value is used in NHN22

# --- Outputing files ---
output_fname = '2022-01-before-ua2-fix.nc'
print(f"output_fname: {output_fname}")

# --- Generate analysis results ---
if to_generate_data:
    output_file = Dataset(output_fname, 'w')
    output_file.createDimension('latitude', nlat // 2 + 1)
    output_file.createDimension('longitude', nlon)
    output_file.createDimension('time', ntimes + 4)
    lats = output_file.createVariable('latitude', np.dtype('float32').char,
                                      ('latitude',))  # Define the coordinate variables
    lons = output_file.createVariable('longitude', np.dtype('float32').char, ('longitude',))
    times = output_file.createVariable('time', np.dtype('int').char, ('time',))
    lats.units = 'degrees_north'
    lons.units = 'degrees_east'
    times.units = time_units
    times.calendar = time_calendar
    lats[:] = ylat[90:]
    lons[:] = xlon
    # times[:] = time_array
    lwa_baro = output_file.createVariable('lwa', np.dtype('float32').char, ('time', 'latitude', 'longitude'))
    lwa_baro.units = 'm/s'
    ua1 = output_file.createVariable('ua1', np.dtype('float32').char, ('time', 'latitude', 'longitude'))
    ua1.units = 'm/s'
    ua2 = output_file.createVariable('ua2', np.dtype('float32').char, ('time', 'latitude', 'longitude'))
    ua2.units = 'm/s'
    ep1 = output_file.createVariable('ep1', np.dtype('float32').char, ('time', 'latitude', 'longitude'))
    ep1.units = 'm/s'
    ep2 = output_file.createVariable('ep2', np.dtype('float32').char, ('time', 'latitude', 'longitude'))
    ep2.units = 'm/s'
    ep3 = output_file.createVariable('ep3', np.dtype('float32').char, ('time', 'latitude', 'longitude'))
    ep3.units = 'm/s'
    ep4 = output_file.createVariable('ep4', np.dtype('float32').char, ('time', 'latitude', 'longitude'))
    ep4.units = 'm/s'

    # --- Compute LWA + fluxes and save the data into netCDF file ---
    for tstep in range(ntimes):

        uu = u_file.variables['u'][tstep, ::-1, ::-1, :].data
        vv = v_file.variables['v'][tstep, ::-1, ::-1, :].data
        tt = t_file.variables['t'][tstep, ::-1, ::-1, :].data

        qgfield = QGFieldNHN22(xlon, ylat, plev, uu, vv, tt, kmax=kmax, dz=dz, eq_boundary_index=eq_boundary_index)

        qgfield.interpolate_fields()

        qgfield.compute_reference_states()

        astarbaro, u_baro, urefbaro, ua1baro, ua2baro, ep1baro, ep2baro, ep3baro, ep4baro, astar1, astar2 = \
            qgfield._compute_lwa_flux_dirinv(
                np.swapaxes(qgfield.qref[:, -nd:], 0, 1),
                np.swapaxes(qgfield.uref[:, -(nd-eq_boundary_index):], 0, 1),
                np.swapaxes(qgfield.ptref[:, -(nd-eq_boundary_index):], 0, 1))

        lwa_baro[tstep, :, :] = np.swapaxes(astarbaro, 0, 1)
        ua1[tstep, :, :] = np.swapaxes(ua1baro, 0, 1)
        ua2[tstep, :, :] = np.swapaxes(ua2baro, 0, 1)
        ep1[tstep, :, :] = np.swapaxes(ep1baro, 0, 1)
        ep2[tstep, :, :] = np.swapaxes(ep2baro, 0, 1)
        ep3[tstep, :, :] = np.swapaxes(ep3baro, 0, 1)
        ep4[tstep, :, :] = np.swapaxes(ep4baro, 0, 1)

        print(f'tstep = {tstep}/{ntimes}.')
    output_file.close()
