import os
import itertools
import numpy as np
import xarray as xr
from falwa.plot_utils import LatLonMapPlotter, HeightLatPlotter

# Local
data_path = f"{os.environ['HOME']}/Dropbox/GitHub/hn2016_falwa/github_data_storage/predigest/output_2023_01.nc"
# Waves
data_path = "/mnt/winds/data/csyhuang/predigest/"
year_range = np.arange(2022, 2023+1)
month_range = np.arange(1, 12+1)
uvt_names = ["u", "v", "t"]
ncfiles = [
    f"{data_path}output_{year}_{month}_{uvt}.nc" for year, month, uvt in itertools.product(year_range, month_range, uvt_names)]

df = xr.open_mfdataset(
    ncfiles,
    decode_times=False)
lat = df.coords['latitude']
lon = df.coords['longitude']
height = np.arange(0, 49000, 1000)
# *** For plotting ***
lon_range = np.arange(-180, 181, 60)
lat_range = np.arange(-90, 91, 30)
lat_lon_map_plotter = LatLonMapPlotter(
    figsize=(6, 4), title_str="ERA5 2019-2023", xgrid=lon, ygrid=lat, xland=[],
    yland=[], lon_range=lon_range, lat_range=lat_range, wspace=0.5, hspace=0.5,
    exclude_equator=True, white_space_width=20)
height_lat_plotter = HeightLatPlotter(
    figsize=(6, 6), title_str="ERA5 2019-2023", xgrid=lat, ygrid=height, xlim=[-85, 85],
    exclude_equator=True, white_space_width=30)

# *** Lat-lon plots ***
lat_lon_map_plotter.plot_and_save_variable(
    variable=df.variables['lwa_baro'].data.mean(axis=0),
    cmap="jet", var_title_str="<LWA>$\cos\phi$", save_path="lwa_baro.png", num_level=30)
lat_lon_map_plotter.plot_and_save_variable(
    variable=df.variables['u_baro'].data.mean(axis=0),
    cmap="jet", var_title_str="<u>$\cos\phi$", save_path="u_baro.png", num_level=30)
height_lat_plotter.plot_and_save_variable(
    variable=df.variables['uref'].data.mean(axis=0),
    cmap="jet", var_title_str="Uref", save_path="uref.png", num_level=30)
height_lat_plotter.plot_and_save_variable(
    variable=df.variables['ubar'].data.mean(axis=0),
    cmap="jet", var_title_str="ubar", save_path="ubar.png", num_level=30)
height_lat_plotter.plot_and_save_variable(
    variable=df.variables['uref'].data.mean(axis=0) - df.variables['ubar'].data.mean(axis=0),
    cmap="jet", var_title_str="$\Delta$ u", save_path="delta_u.png", num_level=30)

print("d")
