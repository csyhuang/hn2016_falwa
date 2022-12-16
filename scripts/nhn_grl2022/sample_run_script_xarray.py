"""
-------------------------------------------------------------------------------------------------------------------
File name: sample_run_script_xarray.py
Author: Clare Huang
Created on: 2022/12/15
Description: Sample script to reproduce plots in Neal et al (2022, GRL) using xarray to read netCDF file
-------------------------------------------------------------------------------------------------------------------
"""
import os
import sys
import numpy as np
from math import pi
import xarray as xr
from hn2016_falwa.oopinterface import QGField
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
data_dir = "grl2021_data/"
to_generate_data = True

# --- Load the zonal wind and QGPV at 240hPa --- #
u_file = xr.open_dataset(data_dir + '2021_06_u.nc')
v_file = xr.open_dataset(data_dir + '2021_06_v.nc')
t_file = xr.open_dataset(data_dir + '2021_06_t.nc')

time_coords = u_file.coords['time']
ntimes = 2  # ntimes = time_coords.values.size
print('Dimension of time: {}'.format(ntimes))

# --- Longitude, latitude and pressure grid ---
xlon = u_file.coords['longitude'].values

# latitude has to be in ascending order
ylat = u_file.coords['latitude'].values
if np.diff(ylat)[0] < 0:
    print('Flip ylat.')
    ylat = ylat[::-1]

# pressure level has to be in descending order (ascending height)
plev = u_file.coords['level'].values
if np.diff(plev)[0]>0:
    print('Flip plev.')
    plev = plev[::-1]

nlon = xlon.size
nlat = ylat.size
nlev = plev.size

# --- Coordinates ---
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
prefactor = np.array([np.exp(-z/hh) for z in height[1:]]).sum()  # integrated sum of density from the level
                                                                 # just above the ground (z=1km) to aloft
npart = nlat                        # number of partitions to construct the equivalent latitude grids
maxits = 100000                     # maximum number of iteration in the SOR solver to solve for reference state
tol = 1.e-5                         # tolerance that define convergence of solution
rjac = 0.95                         # spectral radius of the Jacobi iteration in the SOR solver.
nd = nlat//2+1                      # (one plus) index of latitude grid point with value 0 deg
                                    # This is to be input to fortran code. The index convention is different.
equatorward_bound = 5               # This is new in Neal et al 2022 when we use lat = 5N as equatorward boundary

# --- Compute LWA + fluxes and save the data into netCDF file ---
for tstep in range(ntimes):

    uu = u_file.variables['u'].isel(time=0).values[::-1, ::-1, :]
    vv = v_file.variables['v'].isel(time=0).values[::-1, ::-1, :]
    tt = t_file.variables['t'].isel(time=0).values[::-1, ::-1, :]

    qgfield_object = QGField(xlon, ylat, plev, uu, vv, tt, kmax=kmax, dz=dz, eq_boundary_index=5)

    qgpv_temp, interpolated_u_temp, interpolated_v_temp, interpolated_avort_temp, interpolated_theta_temp, \
    static_stability_n, static_stability_s, tn0, ts0 = qgfield_object._interpolate_field_dirinv()

    qref, uref, tref, fawa, ubar, tbar = qgfield_object._compute_qref_fawa_and_bc()

    astarbaro, ubaro, urefbaro, ua1baro, ua2baro, ep1baro, ep2baro, ep3baro, ep4baro, astar1, astar2 = \
        qgfield_object._compute_lwa_flux_dirinv(qref, uref, tref)

    output = xr.Dataset({
        "qref": (("latitude", "level"), qref[equatorward_bound:, :]),
        "uref": (("latitude", "level"), uref),
        "tref": (("latitude", "level"), tref),
        "astarbaro": (("longitude", "latitude"), astarbaro[:, equatorward_bound:]),
        "ubaro": (("longitude", "latitude"), ubaro[:, equatorward_bound:])
    },
        coords={
            "level": height,
            "latitude": ylat[nd+equatorward_bound-1:],
            "longitude": xlon,
        },
    )
    output.to_netcdf(f"tstep_{tstep}.nc")

    # plt.figure()
    # plt.contourf(xlon, ylat[nd+equatorward_bound-1:], output.variables['astarbaro'].data.swapaxes(0, 1))
    # plt.xlabel('longitude[deg]')
    # plt.ylabel('latitude[deg]')
    # plt.show()

    print(f'tstep = {tstep}/{ntimes}.')
print("Finished.")

