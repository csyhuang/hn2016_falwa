import os
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype
from scipy.interpolate import interp1d
from netCDF4 import Dataset
#import matplotlib.pyplot as plt
import datetime as dt
#get_ipython().run_line_magic('matplotlib', 'inline')
 
from hn2016_falwa.oopinterface import QGField
#from download_a_year import download_data_whole_year



#import sys
#sys.executable

data_dir = ''
d_sim = Dataset(data_dir+'atmos_daily.nc','r')



u = d_sim.variables['ucomp'][:].data
v = d_sim.variables['vcomp'][:].data
t = d_sim.variables['temp'][:].data



time_array = d_sim.variables['time'][:].data
time_units = d_sim.variables['time'].units
time_calendar = d_sim.variables['time'].calendar
ntimes = time_array.shape[0]

print('Dimension of time: {}'.format(time_array.size))



# --- Load grids --- #
xlon = d_sim.variables['lon'][:].data

# latitude has to be in ascending order
ylat = d_sim.variables['lat'][:].data
if np.diff(ylat)[0]<0:
    print('Flip ylat.')
    ylat = ylat[::-1]
    
# pressure level has to be in descending order (ascending height)
plev = d_sim.variables['pfull'][:].data
if np.diff(plev)[0]>0:
    print('Flip plev.')    
    plev = plev[::-1]

nlon = xlon.size
nlat = ylat.size
nlev = plev.size



# --- Parameters and constants --- #
clat = np.cos(np.deg2rad(ylat))          # cosine latitude
p0 = 1000.                               # surface pressure [hPa]
kmax = 30                                # number of grid points for vertical extrapolation (dimension of height)
height = np.arange(0, (kmax-1)*1000 + 1, 1000)         # pseudoheight [m]
p_grid = 1000. * np.exp(-height/7000.)   # pressure level on regular pseudoheight grid
dz = 1000.                          # differential height element
dphi = np.diff(ylat)[0]*pi/180.     # differential latitudinal element
dlambda = np.diff(xlon)[0]*pi/180.  # differential latitudinal element
hh = 7000.                          # scale height
cp = 1004.                          # heat capacity of dry air
rr = 287.                           # gas constant
omega = 7.29e-5                     # rotation rate of the earth
aa = 6.378e+6                       # earth radius
prefactor = 6500.                   # integrated sum of density from ground to aloft
npart = nlat                        # number of partitions to construct the equivalent latitude grids
maxits = 100000                     # maximum number of iteration in the SOR solver to solve for reference state
tol = 1.e-5                         # tolerance that define convergence of solution
rjac = 0.95                         # spectral radius of the Jacobi iteration in the SOR solver.              
jd = nlat//2+1                      # (one plus) index of latitude grid point with value 0 deg
                                    # This is to be input to fortran code. The index convention is different.

# --- Define variables in output file --- #
output_fname = 'atmos_daily_lowab_tropics_lwa_flux.nc'
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
plevs    = p0 * np.exp(-height/hh)
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
adv_flux_f1 = output_file.createVariable('Zonal advective flux F1',dtype('float32').char,('time','latitude','longitude'))
adv_flux_f1.units = 'm**2/s**2'    
adv_flux_f2 = output_file.createVariable('Zonal advective flux F2',dtype('float32').char,('time','latitude','longitude'))
adv_flux_f2.units = 'm**2/s**2'  
adv_flux_f3 = output_file.createVariable('Zonal advective flux F3',dtype('float32').char,('time','latitude','longitude'))
adv_flux_f3.units = 'm**2/s**2'  
adv_flux_conv = output_file.createVariable('Zonal advective flux Convergence -Div(F1+F2+F3)',dtype('float32').char,('time','latitude','longitude'))
adv_flux_conv.units = 'm/s**2'  
divergence_eddy_momentum_flux = output_file.createVariable('Eddy Momentum Flux Divergence',dtype('float32').char,('time','latitude','longitude'))
divergence_eddy_momentum_flux.units = 'm/s**2'  
meridional_heat_flux = output_file.createVariable('Low-level Meridional Heat Flux',dtype('float32').char,('time','latitude','longitude'))
meridional_heat_flux.units = 'm/s**2' 
lwa_baro = output_file.createVariable('lwa_baro',dtype('float32').char,('time','latitude','longitude'))
lwa_baro.units = 'm/s' 
u_baro = output_file.createVariable('u_baro',dtype('float32').char,('time','latitude','longitude'))
u_baro.units = 'm/s'



# --- Compute local wave activity and output the data at a particular latitude --- #
#for tstep in range(ntimes):
    
print('ylat = {}'.format(ylat))
print('plev = {}'.format(plev))
print('nlat = {}'.format(nlat))
print(d_sim)

for tstep in range(ntimes):
    
    uu = d_sim.variables['ucomp'][tstep, ::-1, :, :].data
    vv = d_sim.variables['vcomp'][tstep, ::-1, :, :].data
    tt = d_sim.variables['temp'][tstep, ::-1, :, :].data

    qgfield_object = QGField(xlon, ylat, plev, uu, vv, tt, kmax=kmax)

    qgpv[tstep, :, :, :], interpolated_u[tstep, :, :, :], interpolated_v[tstep, :, :, :], \
        interpolated_theta[tstep, :, :, :], static_stability = qgfield_object.interpolate_fields()
    print('Finished interpolation. Start computing reference state.')

    # plt.contourf(xlon, ylat, interpolated_u[tstep, 10, :, :])
    # plt.colorbar()
    # plt.show()

    qref[tstep, :, (nlat//2):], uref[tstep, :, (nlat//2):], ptref[tstep, :, (nlat//2):] = \
        qgfield_object.compute_reference_states(northern_hemisphere_results_only=True)

    print('Finished computing reference state.')

    adv_flux_f1[tstep, (nlat//2):, :], \
    adv_flux_f2[tstep, (nlat//2):, :], \
    adv_flux_f3[tstep, (nlat//2):, :], \
    adv_flux_conv[tstep, (nlat//2):, :], \
    divergence_eddy_momentum_flux[tstep, (nlat//2):, :], \
    meridional_heat_flux[tstep, (nlat//2):, :], \
    lwa_baro[tstep, (nlat//2):, :], \
    u_baro[tstep, (nlat//2):, :], \
    lwa[tstep, :, (nlat//2):, :] \
        = qgfield_object.compute_lwa_and_barotropic_fluxes(northern_hemisphere_results_only=True)
    
    print(tstep)
    
output_file.close()
print('Output {} timesteps of data to the file {}'.format(ntimes, output_fname))


# --- Delete netCDF files downloaded for that year ---
#    for file in files_downloaded:
#        os.remove(file)
#        print('{} deleted.'.format(file))



