import numpy as np
from math import pi
from netCDF4 import Dataset
from hn2016_falwa.oopinterface import QGField
import datetime as dt
import matplotlib.pyplot as plt

data_dir = "grl2022_data/"

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
jd = nlat//2+1                      # (one plus) index of latitude grid point with value 0 deg
                                    # This is to be input to fortran code. The index convention is different.


# --- Outputing files ---
output_fname = '2021-06-01_to_2021-06-30_output3.nc'
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


print(f'ylat[90:].size = {ylat[90:].size}.')

# --- Set timestamp and pressure level to display ---
tstamp = [dt.datetime(2005,1,23,0,0) + dt.timedelta(seconds=6*3600) * tt for tt in range(ntimes)]
plev_selected = 10 # selected pressure level to display
tstep_selected = 0

# --- Compute LWA and fluxes ---
for tstep in range(ntimes):  # or ntimes

    uu = u_file.variables['u'][tstep, ::-1, ::-1, :].data
    vv = v_file.variables['v'][tstep, ::-1, ::-1, :].data
    tt = t_file.variables['t'][tstep, ::-1, ::-1, :].data

    qgfield_object = QGField(xlon, ylat, plev, uu, vv, tt, kmax=kmax, dz=dz)

    qgpv_temp, interpolated_u_temp, interpolated_v_temp, interpolated_avort_temp, interpolated_theta_temp, \
    static_stability_n, static_stability_s, tn0, ts0 = qgfield_object._interpolate_field_dirinv()

    # plt.contourf(xlon, ylat, np.swapaxes(qgpv_temp[:, :, 40], 0, 1), cmap='rainbow')
    # plt.colorbar()
    # plt.savefig("test_qgpv.jpg")
    # plt.clf()
    # plt.plot(ylat, interpolated_u_temp[:, :, 40].mean(axis=0))
    # # plt.colorbar()
    # plt.savefig("test_zm_u.jpg")
    # plt.clf()
    # plt.contourf(xlon, ylat, np.swapaxes(interpolated_theta_temp[:, :, 20], 0, 1), cmap='rainbow')
    # plt.colorbar()
    # plt.savefig("test_pt.jpg")

    print("Before compute_qref_fawa_and_bc")
    qref, uref, tref, fawa, ubar, tbar = qgfield_object._compute_qref_fawa_and_bc()

    # print(f"qref.shape = {qref.shape}")
    # plt.contourf(np.arange(91), np.arange(0, 48500, 500), np.swapaxes(qref, 0, 1), cmap='rainbow')
    # plt.colorbar()
    # plt.ylabel("height")
    # plt.xlabel("latitude")
    # plt.savefig("qref.png")

    print("After compute_qref_fawa_and_bc")
    astarbaro, ubaro, urefbaro, ua1baro, ua2baro, ep1baro, ep2baro, ep3baro, ep4baro, astar1, astar2 = \
        qgfield_object._compute_lwa_flux_dirinv(qref, uref, tref, fawa, ubar, tbar)

    for k in range(0, 96):
        astar1_nan = np.count_nonzero(astar1[:, 80:, k]>1000)
        astar2_nan = np.count_nonzero(astar2[:, 80:, k]>1000)
        fawa_nan = np.count_nonzero(fawa[80:, k]>1000)
        if astar1_nan + astar2_nan + fawa_nan > 0:
            print(f"k = {k}. nan in astar1 = {astar1_nan}. nan in astar2 = {astar2_nan}. nan in fawa = {fawa_nan}.")


    # print(f"ans2[0].shape = {ans2[0].shape}")
    # print(f"ans2[1].shape = {ans2[1].shape}")
    # print(f"ans2[0] = {ans2[0]}")
    # print(f"ans2[1] = {ans2[1]}")

    # print(f"qref.shape = {qref.shape}")
    # print(f"qref[:, 40] = {qref[:, 40]}")
    # print(f"fawa.shape = {fawa.shape}")
    # print(f"fawa[:, 40] = {fawa[:, 40]}")
    # print(f"tstep = {tstep}")

    # qref, uref, ptref = qgfield_object.compute_reference_states(northern_hemisphere_results_only=False)
    #
    # barotropic_fluxes = qgfield_object.compute_lwa_and_barotropic_fluxes(northern_hemisphere_results_only=False)
    #
    lwa_baro[tstep, :, :] = np.swapaxes(astarbaro, 0, 1)
    ua1[tstep, :, :] = np.swapaxes(ua1baro, 0, 1)
    ua2[tstep, :, :] = np.swapaxes(ua2baro, 0, 1)
    ep1[tstep, :, :] = np.swapaxes(ep1baro, 0, 1)
    ep2[tstep, :, :] = np.swapaxes(ep2baro, 0, 1)
    ep3[tstep, :, :] = np.swapaxes(ep3baro, 0, 1)
    ep4[tstep, :, :] = np.swapaxes(ep4baro, 0, 1)

    print(f'tstep = {tstep}/{ntimes}.')

# output_file.close()
