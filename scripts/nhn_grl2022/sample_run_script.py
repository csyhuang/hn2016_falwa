"""
-------------------------------------------------------------------------------------------------------------------
File name: sample_run_script.py
Author: Clare Huang
Created on: 2022/3/16
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
from graph_plot_module import plot_figure1a, plot_figure1b, plot_figure1c, plot_figure1d_2a, plot_figure3_and_S1, \
    plot_figure3e, plot_figure3f, plot_figure4, plot_figure5

data_dir = "grl2021_data/"
to_generate_data = True  # Set this to False if wave activity and fluxes are computed and stored in netCDF file already

# --- Load the zonal wind and QGPV at 240hPa --- #
u_file = Dataset(data_dir + '2021_06_u.nc', mode='r')
v_file = Dataset(data_dir + '2021_06_v.nc', mode='r')
t_file = Dataset(data_dir + '2021_06_t.nc', mode='r')

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
output_fname = '2021-06-01_to_2021-06-30_output.nc'
print(f"output_fname: {output_fname}")

# --- Generate analysis results ---
if to_generate_data:
    output_file = Dataset(output_fname, 'w')
    output_file.createDimension('latitude',nlat//2+1)
    output_file.createDimension('longitude',nlon)
    output_file.createDimension('time',ntimes+4)
    lats = output_file.createVariable('latitude',np.dtype('float32').char,('latitude',)) # Define the coordinate variables
    lons = output_file.createVariable('longitude',np.dtype('float32').char,('longitude',))
    times = output_file.createVariable('time',np.dtype('int').char,('time',))
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

# --- Graph plotting for GRL2021 ---

# Location of data files
data_dir = "grl2021_data/"
z_filename = data_dir + "2021_06_z.nc"     # geopotential height
u_filename = data_dir + "2021_06_u.nc"     # u
v_filename = data_dir + "2021_06_v.nc"     # v
t_filename = data_dir + "2021_06_t.nc"     # temperature
t2m_filename = data_dir + "2021_06_2t.nc"     # t2m
mtnlwrf_filename = data_dir + "2021_06_mtnlwrf.nc"     # net OLR
mtnlwrfcs_filename = data_dir + "2021_06_mtnlwrfcs.nc"   # OLR clear sky
tcw_filename = data_dir + "2021_06_tcw.nc"     # total column water (kg/m^2)
tcwv_filename = data_dir + "2021_06_tcwv.nc"   # total column water vapor (kg/m^2)
sp_filename = data_dir + "2021_06_sp.nc"   # sea level pressure (hPa)
lwa_flux_filename = output_fname

# Execute graph plotting functions
plot_figure1a(z_filename, u_filename, v_filename)
plot_figure1b(t_filename)
plot_figure1c(t2m_filename)
plot_figure1d_2a(t_filename)
plot_figure3_and_S1(lwa_flux_filename)
plot_figure3e(mtnlwrf_filename, mtnlwrfcs_filename)
plot_figure3f(tcw_filename, tcwv_filename, sp_filename)
plot_figure4(lwa_flux_filename)
plot_figure5(lwa_flux_filename)
