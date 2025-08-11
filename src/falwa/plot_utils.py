"""
------------------------------------------
File name: plot_utils.py
Author: Clare Huang
"""
import numpy as np
import xarray as xr
from matplotlib import gridspec, pyplot as plt  # Optional dependency
from matplotlib.axes import Axes
from matplotlib.contour import ContourSet
from cartopy import crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from typing import Tuple, Optional, Sequence


def compare_two_fields_on_map(
    field_a: np.ndarray, field_b: np.ndarray, a_title: str, b_title: str, x_coord: np.ndarray, y_coord: np.ndarray,
    title: str, savefig_fname: Optional[str] = 'default.png', diff_factor: Optional[float] = 0.01,
    figsize: Tuple[int, int] = (15, 4), cmap: str = 'rainbow') -> None:
    """Compare two 2D-fields on a map and plot their difference.

    The output plot has 3 columns:
    1. value of field_a,
    2. value of field_b,
    3. absolute difference between field_a and field_b.

    The color scale of this plot can be controlled by the parameter `diff_factor`:
    the color range of the plot will be the maximum value among field_a and
    field_b multiplied by diff_factor. If you want to use the auto-colorscale,
    set diff_factor to None.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    field_a : np.ndarray
        First 2D-field to compare.
    field_b : np.ndarray
        Second 2D-field to compare.
    a_title : str
        Title of the first field.
    b_title : str
        Title of the second field.
    x_coord : np.ndarray
        Array of x-coordinates.
    y_coord : np.ndarray
        Array of y-coordinates.
    title : str
        Main title of the whole plot.
    savefig_fname : str, optional
        Filename of figure saved. Default: "default.png". If you don't want a
        file to be saved, set this to None.
    diff_factor : float, optional
        The color range of the plot will be the maximum value among field_a and
        field_b multiplied by diff_factor. If you want to use the
        auto-colorscale, set diff_factor to None. Default: 0.01.
    figsize : tuple[int, int], optional
        Tuple specifying figure size.
    cmap : str, optional
        Colormap to use for the plots.
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
    if diff_factor:
        levels = np.linspace(0, diff_factor * max([np.abs(cmin), np.abs(cmax)]), 31)
    else:
        levels = None  # type: ignore
    cs3 = map_on_axis(ax3, x_coord, y_coord, np.abs(field_a - field_b),
                      levels=levels,
                      title='Abs difference',
                      colorbar_label='')
    plt.suptitle(title)
    plt.tight_layout()
    if savefig_fname:
        plt.savefig(savefig_fname)
    plt.show()


def map_on_axis(ax_obj: Axes, xlon: np.ndarray, ylat: np.ndarray, field: np.ndarray,
                levels: Optional[Sequence[float]], title: str, colorbar_label: str = 'ua2 (m/s**2)') -> ContourSet:
    """Plot a 2D field on a map axis.

    Parameters
    ----------
    ax_obj : matplotlib.axes.Axes
        The axis object to plot on.
    xlon : np.ndarray
        Longitude coordinates.
    ylat : np.ndarray
        Latitude coordinates.
    field : np.ndarray
        The 2D data field to plot.
    levels : Sequence[float] or None
        Contour levels. If None, levels are chosen automatically.
    title : str
        Title for the subplot.
    colorbar_label : str, optional
        Label for the colorbar.

    Returns
    -------
    matplotlib.contour.ContourSet
        The contour set object.
    """
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
        field_a: np.ndarray, field_b: np.ndarray, a_title: str, b_title: str, x_coord: np.ndarray, y_coord: np.ndarray,
        title: str, savefig_fname: Optional[str] = 'default.png', diff_factor: Optional[float] = 0.01,
        figsize: Tuple[int, int] = (15, 4), cmap: str = 'rainbow') -> None:
    """Compare two 2D-fields and plot their difference.

    The output plot has 3 columns:
    1. value of field_a,
    2. value of field_b,
    3. absolute difference between field_a and field_b.

    The color scale of this plot can be controlled by the parameter `diff_factor`:
    the color range of the plot will be the maximum value among field_a and
    field_b multiplied by diff_factor. If you want to use the auto-colorscale,
    set diff_factor to None.

    .. versionadded:: 0.7.0

    Parameters
    ----------
    field_a : np.ndarray
        First 2D-field to compare.
    field_b : np.ndarray
        Second 2D-field to compare.
    a_title : str
        Title of the first field.
    b_title : str
        Title of the second field.
    x_coord : np.ndarray
        Array of x-coordinates.
    y_coord : np.ndarray
        Array of y-coordinates.
    title : str
        Main title of the whole plot.
    savefig_fname : str, optional
        Filename of figure saved. Default: "default.png". If you don't want a
        file to be saved, set this to None.
    diff_factor : float, optional
        The color range of the plot will be the maximum value among field_a and
        field_b multiplied by diff_factor. If you want to use the
        auto-colorscale, set diff_factor to None. Default: 0.01.
    figsize : tuple[int, int], optional
        Tuple specifying figure size.
    cmap : str, optional
        Colormap to use for the plots.
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


def plot_lon_lat_field(filepath: str, variable_name: str, latitude_name: str = 'latitude',
                       longitude_name: str = 'longitude', tstep: int = 10, zonal_axis: int = 0) -> None:
    """Plot a snapshot of longitude-latitude map from a netCDF file.

    Parameters
    ----------
    filepath : str
        Path to the netCDF file.
    variable_name : str
        Name of the variable to be plotted.
    latitude_name : str, optional
        Name of latitudinal coordinates.
    longitude_name : str, optional
        Name of longitudinal coordinates.
    tstep : int, optional
        Index of timestep to be plotted.
    zonal_axis : int, optional
        Axis of zonal dimension.
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


class LatLonMapPlotter(object):
    """Utility for plotting latitude-longitude maps with Cartopy.

    Parameters
    ----------
    figsize : tuple[int, int]
        Figure size in inches.
    title_str : str
        Title of the plot.
    xgrid : np.ndarray
        Longitude grid points (in deg).
    ygrid : np.ndarray
        Latitude grid points (in deg).
    xland : np.ndarray
        With dimension same as np.meshgrid(xgrid, ygrid).
        Longitude grid points to be masked out.
    yland : np.ndarray
        With dimension same as np.meshgrid(xgrid, ygrid).
        Latitude grid points to be masked out.
    lon_range : np.ndarray
        Longitude ticks/marks displayed on the plot.
    lat_range : np.ndarray
        Latitude ticks/marks displayed on the plot.
    wspace : float, optional
        Width space size.
    hspace : float, optional
        Height space size.
    coastlines_alpha : float, optional
        Degree of transparency of coastlines (in percent).
    exclude_equator : bool, optional
        Whether to exclude equator or not.
    white_space_width : float, optional
        The width of white space centered at equator if exclude_equator is True.
    x_axis_title : str, optional
        Label of the x-axis.
    y_axis_title : str, optional
        Label of the y-axis.
    """
    def __init__(self, figsize: Tuple[int, int], title_str: str, xgrid: np.ndarray, ygrid: np.ndarray,
                 xland: np.ndarray, yland: np.ndarray, lon_range: np.ndarray, lat_range: np.ndarray,
                 wspace: float = 0.3, hspace: float = 0.3, coastlines_alpha: float = 0.7,
                 exclude_equator: bool = True, white_space_width: float = 20,
                 x_axis_title: str = "Longitude[deg]", y_axis_title: str = "Latitude[deg]"):
        self._figsize = figsize
        self._title_str = title_str
        self._xgrid = xgrid
        self._ygrid = ygrid
        self._xland = xland
        self._yland = yland
        self._lon_range = lon_range
        self._lat_range = lat_range
        self._wspace = wspace
        self._hspace = hspace
        self._coastlines_alpha = coastlines_alpha
        self._exclude_equator = exclude_equator
        self._white_space_width = white_space_width
        self._x_axis_title = x_axis_title
        self._y_axis_title = y_axis_title

    def plot_and_save_variable(self, variable: np.ndarray, cmap: str, var_title_str: str,
                               save_path: Optional[str] = "figure.png", num_level: int = 30) -> None:
        """Plot and save a variable on a lat-lon map.

        Parameters
        ----------
        variable : np.ndarray
            The 2D data to plot.
        cmap : str
            The colormap to use.
        var_title_str : str
            The title for the variable being plotted.
        save_path : str or None, optional
            Path to save the figure. If None, figure is not saved.
            Defaults to "figure.png".
        num_level : int, optional
            Number of contour levels.
        """
        from cartopy import crs as ccrs
        fig = plt.figure(figsize=self._figsize)
        spec = gridspec.GridSpec(
            ncols=1, nrows=1, wspace=self._wspace, hspace=self._hspace)
        ax = fig.add_subplot(spec[0], projection=ccrs.PlateCarree())
        ax.coastlines(color='black', alpha=self._coastlines_alpha)
        ax.set_aspect('auto', adjustable=None)
        mesh_x, mesh_y = np.meshgrid(self._xgrid, self._ygrid)
        main_fig = ax.contourf(
            mesh_x, mesh_y,
            variable,
            num_level,
            cmap=cmap, transform=ccrs.PlateCarree(), transform_first=True)
        ax.scatter(self._xgrid[self._xland], self._ygrid[self._yland], s=1, c='gray')
        if self._exclude_equator:
            ax.axhline(y=0, c='w', lw=self._white_space_width)
        ax.set_xticks(self._lon_range, crs=ccrs.PlateCarree())
        ax.set_yticks(self._lat_range, crs=ccrs.PlateCarree())
        fig.colorbar(main_fig, ax=ax)
        ax.set_title(f"{self._title_str}\n{var_title_str}")
        ax.set_xlabel(self._x_axis_title)
        ax.set_ylabel(self._y_axis_title)
        if save_path:  # input save path is not None
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()


class HeightLatPlotter(object):
    """Utility for plotting Height-latitude plot with `matplotlib.pyplot.contourf`.

    Parameters
    ----------
    figsize : tuple[int, int]
        Figure size in inches.
    title_str : str
        Title of the plot.
    xgrid : np.ndarray
        Latitude grid points (in deg).
    ygrid : np.ndarray
        Height/pressure grid points.
    xlim : tuple[float, float]
        Lower and Upper bound of x-axis.
    exclude_equator : bool, optional
        Whether to exclude equator or not.
    white_space_width : float, optional
        The width of white space centered at equator if exclude_equator is True.
    x_axis_title : str, optional
        Label of the x-axis.
    y_axis_title : str, optional
        Label of the y-axis.
    """
    def __init__(self, figsize: Tuple[int, int], title_str: str, xgrid: np.ndarray, ygrid: np.ndarray,
                 xlim: Tuple[float, float], exclude_equator: bool = True, white_space_width: float = 20,
                 x_axis_title: str = "Latitude[deg]", y_axis_title: str = "Pseudoheight[m]"):
        self._figsize = figsize
        self._title_str = title_str
        self._xgrid = xgrid
        self._ygrid = ygrid
        self._xlim = xlim  # [-80, 80]
        self._exclude_equator = exclude_equator
        self._white_space_width = white_space_width
        self._x_axis_title = x_axis_title
        self._y_axis_title = y_axis_title

    def plot_and_save_variable(self, variable: np.ndarray, cmap: str, var_title_str: str,
                               save_path: str, num_level: int = 30) -> None:
        """Plot and save a variable on a height-latitude plot.

        Parameters
        ----------
        variable : np.ndarray
            The 2D data to plot.
        cmap : str
            The colormap to use.
        var_title_str : str
            The title for the variable being plotted.
        save_path : str
            Path to save the figure.
        num_level : int, optional
            Number of contour levels.
        """
        fig = plt.figure(figsize=self._figsize)
        spec = gridspec.GridSpec(ncols=1, nrows=1)
        ax = fig.add_subplot(spec[0])
        # *** Zonal mean U ***
        main_fig = ax.contourf(
            self._xgrid, self._ygrid,
            variable,
            num_level,
            cmap=cmap)
        ax.set_xlabel(self._x_axis_title)
        ax.set_ylabel(self._y_axis_title)
        fig.colorbar(main_fig, ax=ax)
        if self._exclude_equator:
            ax.axvline(x=0, c='w', lw=self._white_space_width)
        ax.set_title(f"{self._title_str}\n{var_title_str}")
        ax.set_xlim(self._xlim)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Finished saving: {save_path}")
