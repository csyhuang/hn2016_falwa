"""
------------------------------------------
File name: plot_utils.py
Author: Clare Huang
"""
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt  # Optional dependency
from cartopy import crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def compare_two_fields_on_map(
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
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree(180))
    cs1 = map_on_axis(ax1, x_coord, y_coord, field_a, levels=np.linspace(cmin, cmax, 31), title=a_title,
                      colorbar_label='')
    ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree(180))
    cs2 = map_on_axis(ax2, x_coord, y_coord, field_b, levels=np.linspace(cmin, cmax, 31), title=b_title,
                      colorbar_label='')
    ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree(180))
    cs3 = map_on_axis(ax3, x_coord, y_coord, np.abs(field_a - field_b),
                      levels=np.linspace(0, diff_factor * max([np.abs(cmin), np.abs(cmax)]), 31),
                      title=f'Abs difference',
                      colorbar_label='')
    plt.suptitle(title)
    plt.tight_layout()
    if savefig_fname:
        plt.savefig(savefig_fname)
    plt.show()


def map_on_axis(ax_obj, xlon, ylat, field, levels, title, colorbar_label='ua2 (m/s**2)'):
    # ax_obj.set_extent([0, 360, ylat.min(), ylat.max()], ccrs.PlateCarree())
    ax_obj.coastlines(color='black', alpha=1)
    ax_obj.set_aspect('auto')
    ax_obj.set_xticks(np.arange(0, 361, 60), crs=ccrs.PlateCarree())
    ax_obj.set_yticks(np.arange(ylat.min(), ylat.max() + 1, 10), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax_obj.xaxis.set_major_formatter(lon_formatter)
    ax_obj.yaxis.set_major_formatter(lat_formatter)
    ott = ax_obj.contourf(xlon, ylat, field, levels, transform=ccrs.PlateCarree(), cmap='rainbow')
    ax_obj.set_title(title)
    plt.colorbar(ott, ax=ax_obj, label=colorbar_label)
    return ott


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
    cbar1 = plt.colorbar(cs1, ax=ax1)
    cs2 = ax2.contourf(x_coord, y_coord, field_b, np.linspace(cmin, cmax, 31), cmap=cmap)
    ax2.set_title(b_title)
    cbar2 = plt.colorbar(cs2, ax=ax2)
    if diff_factor:
        cs3 = ax3.contourf(x_coord, y_coord, np.abs(field_a-field_b),
                           np.linspace(0, diff_factor * max([np.abs(cmin), np.abs(cmax)]), 31), cmap=cmap)
    else:
        cs3 = ax3.contourf(x_coord, y_coord, np.abs(field_a - field_b), cmap=cmap)
    ax3.set_title(f'Abs difference')
    cbar3 = plt.colorbar(cs3, ax=ax3)
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
