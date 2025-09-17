#==============================================================================
# By Dr. Sandro W. Lubis, May 2024
# Computes the contribution of diabatic heating to the diabatic source/sink term 
# in the Local Wave Activity (LWA) budget.
# Earth Scientist, PNNL
# Input: daily-mean MERRA-2 reanalysis data on pressure levels
# Variables:
#   - U (zonal wind, m/s)       
#   - V (meridional wind, m/s)
#   - T (air temperature, K)
#   - dT/dt (diabatic heating rate, K/s)
#==============================================================================

from netCDF4 import Dataset
import numpy as np
from falwa.oopinterface import QGFieldNHN22

# Constants
p0 = 1000.0  # Reference pressure (hPa)
hh = 7000.0  # Scale height (m)
kmax = 49    # Number of vertical levels
height = np.arange(0, kmax) * 1000.0  # Vertical height array (m)

# Define years to process
years = range(1980, 2021)  # Adjust as needed

for i in years:
    # Open input files for the current year
    u_file = Dataset(f"/pscratch/sd/s/slubis/MERRA2/U/intp/U.{i}_plev.nc", mode='r')
    v_file = Dataset(f"/pscratch/sd/s/slubis/MERRA2/V/intp/V.{i}_plev.nc", mode='r')
    t_file = Dataset(f"/pscratch/sd/s/slubis/MERRA2/T/intp/T.{i}_plev.nc", mode='r')
    dtdt_file = Dataset(f"/pscratch/sd/s/slubis/MERRA2/DTDTMST/intp/DTDTMST.{i}_plev.nc", mode='r')

    # Extract time and spatial coordinates
    time_array = u_file.variables['time'][:]
    time_units = u_file.variables['time'].units
    time_calendar = u_file.variables['time'].calendar if 'calendar' in u_file.variables['time'].ncattrs() else 'standard'
    ntimes = time_array.shape[0]

    xlon = u_file.variables['lon'][:]

    # Latitude must be in ascending order
    ylat = u_file.variables['lat'][:]
    if np.diff(ylat)[0] < 0:
        print('Flip ylat.')
        ylat = ylat[::-1]

    # Pressure levels must be in descending order (ascending height)
    plev = u_file.variables['level'][:] #/ 100.0  # Convert to hPa
    if np.diff(plev)[0] > 0:
        print('Flip plev.')
        plev = plev[::-1]

    nlon = xlon.size
    nlat = ylat.size
    nlev = plev.size

    # Create output file
    output_fname = f"ncforce_3D.{i}.nc"
    output_file = Dataset(output_fname, 'w')

    # Define dimensions
    output_file.createDimension('levelist', kmax)
    output_file.createDimension('latitude', nlat)
    output_file.createDimension('longitude', nlon)
    output_file.createDimension('time', ntimes)

    # Define variables
    plevs = output_file.createVariable('levelist', 'f4', ('levelist',))
    lats = output_file.createVariable('latitude', 'f4', ('latitude',))
    lons = output_file.createVariable('longitude', 'f4', ('longitude',))
    times = output_file.createVariable('time', 'i4', ('time',))

    ncforce_3d = output_file.createVariable(
        'ncforce_3D', 'f4', ('time', 'levelist', 'latitude', 'longitude'), zlib=True
    )

    # Add attributes
    plevs.units = 'hPa'
    lats.units = 'degrees_north'
    lons.units = 'degrees_east'
    times.units = time_units
    times.calendar = time_calendar
    ncforce_3d.units = 'm/s2'

    # Assign coordinate values
    plevs[:] = p0 * np.exp(-height / hh)
    lats[:] = ylat
    lons[:] = xlon
    times[:] = time_array

    # Process and compute ncforce_3D for each timestep
    for tstep in range(ntimes):
        uu = u_file.variables['U'][tstep, :, :, :]
        vv = v_file.variables['V'][tstep, :, :, :]
        tt = t_file.variables['T'][tstep, :, :, :]
        dtdt = dtdt_file.variables['DTDTMST'][tstep, :, :, :]

        # Initialize QGFieldNHN22 object
        qgfield = QGFieldNHN22(
            xlon, ylat, plev, uu, vv, tt,
            northern_hemisphere_results_only=False, eq_boundary_index=5
        )
        qgfield.interpolate_fields(return_named_tuple=False)
        qgfield.compute_reference_states(return_named_tuple=False)

        # Compute ncforce_3D
        ncforce = qgfield.compute_ncforce_from_heating_rate(heating_rate=dtdt)
        qgfield.compute_layerwise_lwa_fluxes(ncforce=ncforce)

        # Save ncforce_3D for this timestep
        ncforce_3d[tstep, :, :, :] = np.swapaxes(
            qgfield._layerwise_flux_terms_storage.ncforce, 0, 2
        )

    # Close the output file (input files remain open until the next iteration)
    output_file.close()

    print(f"Saved ncforce_3D for {i} to {output_fname}")

