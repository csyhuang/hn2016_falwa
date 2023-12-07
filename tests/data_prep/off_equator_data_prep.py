import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


xlon = np.arange(0, 360, 1.25)
ylat = np.linspace(-90, 90, 192, endpoint=True)
plev = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
    750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250,
    225, 200, 175, 150, 125, 100,  70,  50,  30,  20,  10,
    7, 5, 3, 2, 1]

test_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data"
output_netcdf_file = test_data_dir + "/offgrid_input_uvt_data.nc"

u_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-u.nc").interp(
    {"longitude": xlon, "latitude": ylat, "level": plev}, method="linear", kwargs={"fill_value": "extrapolate"})
v_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-v.nc").interp(
    {"longitude": xlon, "latitude": ylat, "level": plev}, method="linear", kwargs={"fill_value": "extrapolate"})
t_file = xr.open_dataset(f"{test_data_dir}/2005-01-23-0000-t.nc").interp(
    {"longitude": xlon, "latitude": ylat, "level": plev}, method="linear", kwargs={"fill_value": "extrapolate"})

ds = xr.Dataset({
    "u": (("level", "latitude", "longitude"), u_file.u.data),
    "v": (("level", "latitude", "longitude"), v_file.v.data),
    "t": (("level", "latitude", "longitude"), t_file.t.data)},
    coords={
        "level": plev,
        "latitude": ylat,
        "longitude": xlon})
ds.to_netcdf(output_netcdf_file)
