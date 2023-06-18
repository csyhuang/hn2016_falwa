"""
------------------------------------------
File name: plot_utils.py
Author: Clare Huang
"""
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt  # Optional dependency


def compare_two_fields(
    field_a, field_b, a_title, b_title, x_coord, y_coord, title, savefig_fname='default.png',
        diff_factor=0.01, figsize=(15, 4), cmap='rainbow') -> None:

    """
    A handy utility to compare the difference between two 2D-fields and plot their difference. The output plot
    has 3 columns:

    1. value of field_a,

    2. value of field_b,

    3. absolute difference between field_a and field_b.

    The color scale of this plot can be controlled by the parameter *diff_factor*: the color range of the plot will be
    the maximum value among field_a and field_b multiplied by diff_factor. If you want to use the auto-colorscale,
    set diff_factor to None.

    .. versionadded:: 0.7.0

    Parameters
    ----------
        field_a : np.ndarray
            First 2D-field to compare
        field_b : np.ndarray
            Second 2D-field to compare
        a_title :str
            Title of the first field
        b_title :str
            Title of the second field
        x_coord : np.array
            array of x-coordinates
        y_coord : np.array
            array of y-coordinates
        title : str
            Main title of the whole plot
        savefig_fname : str
            Filename of figure saved. Default: "default.png". If you don't want a file to be
            saved, set this to None.
        diff_factor : float
            The color range of the plot will be the maximum value among field_a and field_b
            multiplied by diff_factor. If you want to use the auto-colorscale, set diff_factor to None. Default: 0.01.
        figsize : Tuple[int, int]
            tuple specifying figure size
    """

    cmin = np.min([np.amin(field_a), np.amin(field_b)])
    cmax = np.max([np.amax(field_a), np.amax(field_b)])
    print(f"cmin = {cmin}, cmax = {cmax}")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    cs1 = ax1.contourf(x_coord, y_coord, field_a, np.linspace(cmin, cmax, 31), cmap=cmap)
    ax1.set_title(a_title)
    cbar1 = fig.colorbar(cs1)
    cs2 = ax2.contourf(x_coord, y_coord, field_b, np.linspace(cmin, cmax, 31), cmap=cmap)
    ax2.set_title(b_title)
    cbar2 = fig.colorbar(cs2)
    if diff_factor:
        cs3 = ax3.contourf(x_coord, y_coord, np.abs(field_a-field_b),
                           np.linspace(0, diff_factor * max([np.abs(cmin), np.abs(cmax)]), 31), cmap=cmap)
    else:
        cs3 = ax3.contourf(x_coord, y_coord, np.abs(field_a - field_b), cmap=cmap)
    ax3.set_title(f'Abs difference')
    cbar3 = fig.colorbar(cs3)
    plt.suptitle(title)
    plt.tight_layout()
    if savefig_fname:
        plt.savefig(savefig_fname)
    plt.show()


def plot_lon_lat_field(filepath, variable_name, latitude_name='latitude', longitude_name='longitude',
                       tstep=10, zonal_axis=0) -> None:
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
