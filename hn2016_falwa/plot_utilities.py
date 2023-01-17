"""
------------------------------------------
File name: plot_utilities.py
Author: Clare Huang
"""
import xarray as xr
from matplotlib import pyplot as plt


def plot_lon_lat_field(filepath, variable_name, latitude_name='latitude', longitude_name='longitude',
                       tstep=10, zonal_axis=0):
    """
    Plot a snapshot of longitude-latitude map from a netCDF file.

    Parameters
    ----------
    filepath : str
        path to the netCDF file
    variable_name : str
        name of the variable to be plotted
    latitude_name : str
        name of latitudinal coordinates
    longitude_name : str
        name of latitudinal coordinates
    tstep : int
        index of timestep to be plotted
    zonal_axis : int
        axis of zonal dimension
    """
    file_handle = xr.open_dataset(filepath)
    field = file_handle.variables[variable_name].isel(time=tstep).values
    print(f"Zonal mean of the field:\n{field.mean(axis=zonal_axis)}.")

    # plot
    plt.contourf(
        file_handle.coords[longitude_name],
        file_handle.coords[latitude_name],
        file_handle.variables[variable_name].isel(time=tstep))
    plt.show()
