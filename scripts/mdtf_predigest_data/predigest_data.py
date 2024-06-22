import os
import calendar
import json
import datetime
import numpy as np
import argparse
from netCDF4 import Dataset
from falwa.oopinterface import QGFieldNH18

# *** Path to data ***
with open('path.json', "r") as f:
    paths = json.load(f)
vol1_loc = paths['vol1']  # 1979-2018
vol2_loc = paths['vol2']  # 2019-2023

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, help="year")
    parser.add_argument("month", type=int, help="month")
    args = parser.parse_args()
    print(f"args = {args}")

    year = args.year     # 2023
    month = args.month   # 1
    total_tsteps = calendar.monthrange(year, month)[1] * 4

    # u_path = f"{os.environ['HOME']}/Dropbox/GitHub/hn2016_falwa/notebooks/nh2018_science/2005-01-23_to_2005-01-30_u.nc"
    # v_path = f"{os.environ['HOME']}/Dropbox/GitHub/hn2016_falwa/notebooks/nh2018_science/2005-01-23_to_2005-01-30_v.nc"
    # t_path = f"{os.environ['HOME']}/Dropbox/GitHub/hn2016_falwa/notebooks/nh2018_science/2005-01-23_to_2005-01-30_t.nc"

    u_path = f"{vol1_loc}{year}/{year}_{month:02d}_u.nc"
    v_path = f"{vol1_loc}{year}/{year}_{month:02d}_v.nc"
    t_path = f"{vol1_loc}{year}/{year}_{month:02d}_t.nc"
    output_dir = "/mnt/winds/data/csyhuang/predigest/"
    # --- parameters ---
    kmax = 49
    output_filename = f"{output_dir}output_{year}_{month:02d}.nc"

    # --- Load the zonal wind and QGPV at 240hPa --- #
    u_file = Dataset(u_path, mode='r')
    v_file = Dataset(v_path, mode='r')
    t_file = Dataset(t_path, mode='r')

    xlon = u_file.variables['longitude'][:]
    ylat = u_file.variables['latitude'][::-1]
    plev = u_file.variables['level'][::-1]
    time_array = u_file.variables['time'][:]
    time_units = u_file.variables['time'].units
    time_calendar = u_file.variables['time'].calendar
    ntimes = time_array.shape[0]

    output_file = Dataset(output_filename, 'w')
    output_file.createDimension('height', kmax)
    output_file.createDimension('latitude', ylat.size)
    output_file.createDimension('longitude', xlon.size)
    output_file.createDimension('time', total_tsteps)
    height = output_file.createVariable(
        'height', np.dtype('float32').char,
        ('height',))  # Define the coordinate variables
    lats = output_file.createVariable(
        'latitude', np.dtype('float32').char,
        ('latitude',))  # Define the coordinate variables
    lons = output_file.createVariable('longitude', np.dtype('float32').char, ('longitude',))
    times = output_file.createVariable('time', np.dtype('int').char, ('time',))
    lats.units = 'degrees_north'
    lons.units = 'degrees_east'
    times.units = time_units
    times.calendar = time_calendar
    lats[:] = ylat
    lons[:] = xlon
    # times[:] = time_array
    lwa_baro = output_file.createVariable('lwa_baro', np.dtype('float32').char, ('time', 'latitude', 'longitude'))
    lwa_baro.units = 'm/s'
    u_baro = output_file.createVariable('u_baro', np.dtype('float32').char, ('time', 'latitude', 'longitude'))
    u_baro.units = 'm/s'
    fawa = output_file.createVariable('fawa', np.dtype('float32').char, ('time', 'height', 'latitude'))
    fawa.units = 'm/s'
    uref = output_file.createVariable('uref', np.dtype('float32').char, ('time', 'height', 'latitude'))
    uref.units = 'm/s'
    ubar = output_file.createVariable('ubar', np.dtype('float32').char, ('time', 'height', 'latitude'))
    ubar.units = 'm/s'

    print(f"Start computing for year = {year}, month = {month}")
    for tstep in range(total_tsteps):
        uu = u_file.variables['u'][tstep, ::-1, ::-1, :].data
        vv = v_file.variables['v'][tstep, ::-1, ::-1, :].data
        tt = t_file.variables['t'][tstep, ::-1, ::-1, :].data

        qgfield = QGFieldNH18(xlon, ylat, plev, uu, vv, tt, kmax=kmax, dz=1000)
        qgfield.interpolate_fields()
        qgfield.compute_reference_states()
        qgfield.compute_lwa_and_barotropic_fluxes()

        lwa_baro[tstep, :, :] = qgfield.lwa_baro
        u_baro[tstep, :, :] = qgfield.u_baro
        fawa[tstep, :, :] = qgfield.lwa.mean(axis=-1)
        uref[tstep, :, :] = qgfield.uref
        ubar[tstep, :, :] = qgfield.interpolated_u.mean(axis=-1)

        print(f"{datetime.datetime.now()}: Finished output to {output_filename}\ntsteps = {tstep}/{total_tsteps}")
    output_file.close()
    print(f"Close output file {output_filename}")
