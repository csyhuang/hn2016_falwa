import datetime
import os
import itertools
import numpy as np
import xarray as xr
from falwa.plot_utils import LatLonMapPlotter, HeightLatPlotter
from falwa.stat_utils import calculate_covariance

# *** Local ***
# data_path = f"{os.environ['HOME']}/Dropbox/GitHub/hn2016_falwa/github_data_storage/predigest/output_2023_01.nc"
# Waves
data_path = "/mnt/winds/data/csyhuang/predigest/"
start_year = 2003
end_year = 2023
year_range = np.arange(start_year, end_year+1)
season_to_month_range = {
    "DJF": [1, 2, 12],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11]}

for season, month_range in season_to_month_range.items():
    start_time = datetime.datetime.now()
    print(f"Start computing digested data for season {season}. Time now is {start_time}.")
    ncfiles = [
        f"{data_path}output_{year}_{month:02d}.nc" for year, month in itertools.product(year_range, month_range)]
    # *** Local ***
    # ncfiles = [data_path]
    print(f"ncfiles to open:\n{ncfiles}")
    df = xr.open_mfdataset(
        ncfiles,
        decode_times=False)
    lat = df.coords['latitude']
    lon = df.coords['longitude']
    height = np.arange(0, 49000, 1000)
    # *** For plotting ***
    lon_range = np.arange(-180, 181, 60)
    lat_range = np.arange(-90, 91, 30)
    title_str = f"ERA5 {start_year}-{end_year} {season}"
    filename_prefix = f"ERA5_{start_year}-{end_year}_{season}"

    # *** Computation ***
    print("Start computing covariance.")
    lwa_u_covariance = calculate_covariance(df.variables['lwa_baro'].data, df.variables['u_baro'])
    print(f"Finished computing covariance.\n{lwa_u_covariance}\nStart plotting.")

    # *** Output predigested data ***
    lwa_u_covariance_da = xr.DataArray(lwa_u_covariance, {"latitude": lat, "longitude": lon})
    lwa_baro_da = xr.DataArray(df.variables['lwa_baro'].data.mean(axis=0), {"latitude": lat, "longitude": lon})
    u_baro_da = xr.DataArray(df.variables['u_baro'].data.mean(axis=0), {"latitude": lat, "longitude": lon})
    uref_da = xr.DataArray(df.variables['uref'].data.mean(axis=0), {"height": height, "latitude": lat})
    ubar_da = xr.DataArray(df.variables['ubar'].data.mean(axis=0), {"height": height, "latitude": lat})
    delta_u_da = xr.DataArray(df.variables['ubar'].data.mean(axis=0) - df.variables['uref'].data.mean(axis=0),
                              {"height": height, "latitude": lat})
    predigest_dataset_filename = f"{filename_prefix}_predigest_dataset.nc"
    print(f"Start outputing predigest_dataset {predigest_dataset_filename}")
    predigest_dataset = xr.Dataset({
        "lwa_u_covariance": lwa_u_covariance_da,
        "lwa_baro": lwa_baro_da,
        "u_baro": u_baro_da,
        "uref": uref_da,
        "ubar": ubar_da,
        "delta_u": delta_u_da})
    predigest_dataset.to_netcdf(predigest_dataset_filename)
    print(f"Finished outputing predigest_dataset {predigest_dataset_filename}")

    # *** Classes for plotting ***
    lat_lon_map_plotter = LatLonMapPlotter(
        figsize=(6, 4), title_str=title_str,
        xgrid=lon, ygrid=lat, xland=[], yland=[],
        lon_range=lon_range, lat_range=lat_range, wspace=0.5, hspace=0.5,
        exclude_equator=True, white_space_width=20)
    height_lat_plotter = HeightLatPlotter(
        figsize=(6, 6), title_str=title_str,
        xgrid=lat, ygrid=height, xlim=[-85, 85],
        exclude_equator=True, white_space_width=30)

    # *** Lat-lon plots ***
    lat_lon_map_plotter.plot_and_save_variable(
        variable=lwa_baro_da.data,
        cmap="jet", var_title_str="<LWA>$\cos\phi$", save_path=f"{filename_prefix}_lwa_baro.png", num_level=30)
    lat_lon_map_plotter.plot_and_save_variable(
        variable=u_baro_da.data,
        cmap="jet", var_title_str="<u>$\cos\phi$", save_path=f"{filename_prefix}_u_baro.png", num_level=30)
    height_lat_plotter.plot_and_save_variable(
        variable=uref_da.data,
        cmap="jet", var_title_str="Uref", save_path=f"{filename_prefix}_uref.png", num_level=30)
    # *** Height-Lat plots ***
    height_lat_plotter.plot_and_save_variable(
        variable=ubar_da.data,
        cmap="jet", var_title_str="ubar", save_path=f"{filename_prefix}_ubar.png", num_level=30)
    height_lat_plotter.plot_and_save_variable(
        variable=delta_u_da.data,
        cmap="jet", var_title_str="$\Delta$ u", save_path=f"{filename_prefix}_delta_u.png", num_level=30)

    lat_lon_map_plotter.plot_and_save_variable(
        variable=lwa_u_covariance_da.data,
        cmap="jet", var_title_str="COV(<LWA>, <U>)",
        save_path=f"{filename_prefix}_lwa_u_baro_cov.png",
        num_level=30)

    print("Done with calculation!")
    end_time = datetime.datetime.now()
    print(f"Finished computing digested data for season {season}. Time now is {end_time}.")
    print(f"Time used = {end_time-start_time}")
