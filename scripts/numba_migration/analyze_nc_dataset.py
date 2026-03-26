#!/usr/bin/env python
"""
Script to analyze netCDF climate data and compute Local Wave Activity (LWA)
and reference states using either NHN22 or NH18 boundary conditions.

Based on:
- Neal et al. (2022) GRL for NHN22 boundary conditions
- Nakamura and Huang (2018) for NH18 boundary conditions

Usage:
    python analyze_nc_dataset.py --u_file <u_file.nc> --v_file <v_file.nc> --t_file <t_file.nc> --output_dir <output_dir>
    
Example:
    python analyze_nc_dataset.py \
        --u_file 2005-01-23_to_2005-01-30_u.nc \
        --v_file 2005-01-23_to_2005-01-30_v.nc \
        --t_file 2005-01-23_to_2005-01-30_t.nc \
        --output_nc_filename lwa_reference_output_NH18.nc \
        --output_dir ./output_plots_ce9e383_NH18 \
        --qgfield_type NH18

    You can now easily read and compare outputs between different runs using xarray:
    ```python
    import xarray as xr
    ds = xr.open_dataset('output_plots_qgpv_translated/lwa_output_20050123_0000.nc')
    ```

"""

import argparse
import datetime as dt
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from falwa.oopinterface import QGFieldNHN22, QGFieldNH18
import falwa

# Mapping of QGField type names to classes
QGFIELD_CLASSES = {
    'NHN22': QGFieldNHN22,
    'NH18': QGFieldNH18,
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze netCDF climate data and compute LWA and reference states.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--u_file', type=str, required=True,
                        help='Path to the netCDF file containing zonal wind (u) data')
    parser.add_argument('--v_file', type=str, required=True,
                        help='Path to the netCDF file containing meridional wind (v) data')
    parser.add_argument('--t_file', type=str, required=True,
                        help='Path to the netCDF file containing temperature (t) data')
    parser.add_argument('--output_nc_filename', type=str, required=True,
                        help='Filename to save netCDF file with LWA and reference states')
    parser.add_argument('--output_dir', type=str, default='./output_plots_qgpv_translated',
                        help='Directory to save output plots (default: ./output_plots_qgpv_translated)')
    parser.add_argument('--qgfield_type', type=str, choices=['NHN22', 'NH18'], default='NHN22',
                        help='QGField class to use for computation (default: NHN22). '
                             'NHN22: Neal et al. (2022) boundary conditions. '
                             'NH18: Huang & Nakamura (2016, 2017) boundary conditions.')
    parser.add_argument('--tstep', type=int, default=0,
                        help='Time step index to analyze (default: 0)')
    parser.add_argument('--plev_idx', type=int, default=10,
                        help='Pressure level index to display for 3D variables (default: 10)')
    parser.add_argument('--eq_boundary_index', type=int, default=3,
                        help='Equatorward boundary index for NHN22 (default: 3, corresponds to 4.5°N for 1.5° resolution)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figures (default: 150)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots (only save netCDF output)')
    parser.add_argument('--no_netcdf', action='store_true',
                        help='Skip saving netCDF output (only generate plots)')
    return parser.parse_args()


def load_data(u_file, v_file, t_file):
    """
    Load the netCDF data files.
    
    Parameters
    ----------
    u_file : str
        Path to zonal wind file
    v_file : str
        Path to meridional wind file
    t_file : str
        Path to temperature file
        
    Returns
    -------
    tuple
        (u_data, v_data, t_data, time_array, ntimes)
    """
    print(f"Loading data files...")
    print(f"  U file: {u_file}")
    print(f"  V file: {v_file}")
    print(f"  T file: {t_file}")
    
    u_data = xr.open_dataset(u_file)
    v_data = xr.open_dataset(v_file)
    t_data = xr.open_dataset(t_file)
    
    ntimes = u_data.valid_time.size
    time_array = u_data.valid_time
    
    print(f"  Number of time steps: {ntimes}")
    
    return u_data, v_data, t_data, time_array, ntimes


def setup_grid(u_data):
    """
    Set up the grid parameters from the data.
    
    Parameters
    ----------
    u_data : xarray.Dataset
        Dataset containing coordinate information
        
    Returns
    -------
    tuple
        (xlon, ylat, plev, height, kmax, dz, hh)
    """
    xlon = u_data.longitude.values
    
    # latitude has to be in ascending order
    ylat = u_data.latitude.values
    if np.diff(ylat)[0] < 0:
        print("  Flipping latitude to ascending order.")
        ylat = ylat[::-1]
    
    # pressure level has to be in descending order (ascending height)
    plev = u_data.pressure_level.values
    if np.diff(plev)[0] > 0:
        print("  Flipping pressure levels to descending order.")
        plev = plev[::-1]
    
    nlon = xlon.size
    nlat = ylat.size
    nlev = plev.size
    
    print(f"  Grid dimensions: {nlon} lon x {nlat} lat x {nlev} levels")
    
    # Vertical grid parameters
    kmax = 49  # number of grid points for vertical extrapolation
    dz = 1000.  # differential height element [m]
    height = np.arange(0, kmax) * dz  # pseudoheight [m]
    hh = 7000.  # scale height [m]
    
    return xlon, ylat, plev, height, kmax, dz, hh


def extract_fields(u_data, v_data, t_data, tstep):
    """
    Extract u, v, t fields for a given time step.
    
    Parameters
    ----------
    u_data, v_data, t_data : xarray.Dataset
        Input datasets
    tstep : int
        Time step index
        
    Returns
    -------
    tuple
        (uu, vv, tt) numpy arrays
    """
    # Extract and flip latitude dimension to match ascending order
    uu = u_data.u.isel(valid_time=tstep).values[:, ::-1, :]
    vv = v_data.v.isel(valid_time=tstep).values[:, ::-1, :]
    tt = t_data.t.isel(valid_time=tstep).values[:, ::-1, :]
    
    return uu, vv, tt


def compute_lwa_and_fluxes(xlon, ylat, plev, uu, vv, tt, eq_boundary_index=3, qgfield_type='NHN22'):
    """
    Compute LWA, reference states, and barotropic fluxes.
    
    Parameters
    ----------
    xlon, ylat, plev : array-like
        Coordinate arrays
    uu, vv, tt : array-like
        Wind and temperature fields
    eq_boundary_index : int
        Equatorward boundary index for NHN22 (ignored for NH18)
    qgfield_type : str
        QGField class to use ('NHN22' or 'NH18')
        
    Returns
    -------
    QGFieldNHN22 or QGFieldNH18
        Object containing all computed quantities
    """
    print(f"  Initializing QGField object ({qgfield_type})...")
    QGFieldClass = QGFIELD_CLASSES[qgfield_type]
    
    # NHN22 supports eq_boundary_index, NH18 does not
    if qgfield_type == 'NHN22':
        qgfield = QGFieldClass(
            xlon, ylat, plev, uu, vv, tt,
            northern_hemisphere_results_only=False,
            eq_boundary_index=eq_boundary_index
        )
    else:
        qgfield = QGFieldClass(
            xlon, ylat, plev, uu, vv, tt,
            northern_hemisphere_results_only=False
        )
    
    print("  Interpolating fields...")
    qgfield.interpolate_fields(return_named_tuple=False)
    
    print("  Computing reference states...")
    qgfield.compute_reference_states(return_named_tuple=False)
    
    print("  Computing LWA and barotropic fluxes...")
    qgfield.compute_lwa_and_barotropic_fluxes(return_named_tuple=False)
    
    return qgfield


def save_to_netcdf(qgfield, xlon, ylat, height, timestamp, output_dir, eq_boundary_index, filename, qgfield_type='NHN22'):
    """
    Save all computed quantities to a netCDF file.
    
    Parameters
    ----------
    qgfield : QGFieldNHN22 or QGFieldNH18
        Object containing computed quantities
    xlon, ylat : array-like
        Coordinate arrays
    height : array-like
        Pseudoheight array
    timestamp : datetime
        Timestamp for the data
    output_dir : str
        Output directory for the netCDF file
    eq_boundary_index : int
        Equatorward boundary index used in computation
    filename : str
        Name of NetCDF file that contains the output
    qgfield_type : str
        QGField class used ('NHN22' or 'NH18')
        
    Returns
    -------
    str
        Path to the saved netCDF file
    """
    print("  Saving computed quantities to netCDF...")
    
    # Create coordinate arrays for the dataset
    # Note: ylat[1:-1] is used for most variables as the boundary points are excluded
    ylat_interior = ylat[1:-1]
    height_interior = height[1:-1]
    
    # Create the xarray Dataset
    ds = xr.Dataset(
        coords={
            'longitude': ('longitude', xlon, {'units': 'degrees_east', 'long_name': 'Longitude'}),
            'latitude': ('latitude', ylat_interior, {'units': 'degrees_north', 'long_name': 'Latitude'}),
            'latitude_full': ('latitude_full', ylat, {'units': 'degrees_north', 'long_name': 'Latitude (full grid)'}),
            'height': ('height', height, {'units': 'm', 'long_name': 'Pseudoheight'}),
            'height_interior': ('height_interior', height_interior, {'units': 'm', 'long_name': 'Pseudoheight (interior)'}),
            'time': timestamp,
        }
    )
    
    # === 3D Variables (height x latitude x longitude) ===
    # These are on the interpolated vertical grid
    kmax = height.size
    nlat_interior = ylat_interior.size
    
    ds['qgpv'] = xr.DataArray(
        qgfield.qgpv[:, 1:-1, :],
        dims=['height', 'latitude', 'longitude'],
        attrs={'units': 's^-1', 'long_name': 'Quasigeostrophic Potential Vorticity'}
    )
    
    ds['lwa'] = xr.DataArray(
        qgfield.lwa[:, 1:-1, :],
        dims=['height', 'latitude', 'longitude'],
        attrs={'units': 'm s^-1', 'long_name': 'Local Wave Activity'}
    )
    
    ds['interpolated_u'] = xr.DataArray(
        qgfield.interpolated_u[:, 1:-1, :],
        dims=['height', 'latitude', 'longitude'],
        attrs={'units': 'm s^-1', 'long_name': 'Interpolated Zonal Wind'}
    )
    
    ds['interpolated_v'] = xr.DataArray(
        qgfield.interpolated_v[:, 1:-1, :],
        dims=['height', 'latitude', 'longitude'],
        attrs={'units': 'm s^-1', 'long_name': 'Interpolated Meridional Wind'}
    )
    
    # === Reference States (height x latitude) ===
    ds['qref'] = xr.DataArray(
        qgfield.qref[1:-1, 1:-1],
        dims=['height_interior', 'latitude'],
        attrs={'units': 's^-1', 'long_name': 'Reference QGPV'}
    )
    
    ds['uref'] = xr.DataArray(
        qgfield.uref[1:-1, 1:-1],
        dims=['height_interior', 'latitude'],
        attrs={'units': 'm s^-1', 'long_name': 'Reference Zonal Wind'}
    )
    
    ds['ptref'] = xr.DataArray(
        qgfield.ptref[1:-1, 1:-1],
        dims=['height_interior', 'latitude'],
        attrs={'units': 'K', 'long_name': 'Reference Potential Temperature'}
    )
    
    # === Barotropic (vertically averaged) quantities (latitude x longitude) ===
    ds['u_baro'] = xr.DataArray(
        qgfield.u_baro[1:-1, :],
        dims=['latitude', 'longitude'],
        attrs={'units': 'm s^-1', 'long_name': 'Barotropic Zonal Wind'}
    )
    
    ds['lwa_baro'] = xr.DataArray(
        qgfield.lwa_baro[1:-1, :],
        dims=['latitude', 'longitude'],
        attrs={'units': 'm s^-1', 'long_name': 'Barotropic Local Wave Activity'}
    )
    
    ds['adv_flux_f1'] = xr.DataArray(
        qgfield.adv_flux_f1[1:-1, :],
        dims=['latitude', 'longitude'],
        attrs={'units': 'm^2 s^-2', 'long_name': 'Advective Flux F1'}
    )
    
    ds['adv_flux_f2'] = xr.DataArray(
        qgfield.adv_flux_f2[1:-1, :],
        dims=['latitude', 'longitude'],
        attrs={'units': 'm^2 s^-2', 'long_name': 'Advective Flux F2'}
    )
    
    ds['adv_flux_f3'] = xr.DataArray(
        qgfield.adv_flux_f3[1:-1, :],
        dims=['latitude', 'longitude'],
        attrs={'units': 'm^2 s^-2', 'long_name': 'Advective Flux F3'}
    )
    
    ds['convergence_zonal_advective_flux'] = xr.DataArray(
        qgfield.convergence_zonal_advective_flux[1:-1, :],
        dims=['latitude', 'longitude'],
        attrs={'units': 'm s^-2', 'long_name': 'Advective Flux Convergence -Div(F1+F2+F3)'}
    )
    
    ds['divergence_eddy_momentum_flux'] = xr.DataArray(
        qgfield.divergence_eddy_momentum_flux[1:-1, :],
        dims=['latitude', 'longitude'],
        attrs={'units': 'm s^-2', 'long_name': 'Divergence of Eddy Momentum Flux'}
    )
    
    ds['meridional_heat_flux'] = xr.DataArray(
        qgfield.meridional_heat_flux[1:-1, :],
        dims=['latitude', 'longitude'],
        attrs={'units': 'K m s^-1', 'long_name': 'Meridional Heat Flux'}
    )
    
    ds['flux_vector_lambda_baro'] = xr.DataArray(
        qgfield.flux_vector_lambda_baro[1:-1, :],
        dims=['latitude', 'longitude'],
        attrs={'units': 'm^2 s^-2', 'long_name': 'Barotropic Flux Vector (Lambda component)'}
    )
    
    ds['flux_vector_phi_baro'] = xr.DataArray(
        qgfield.flux_vector_phi_baro[1:-1, :],
        dims=['latitude', 'longitude'],
        attrs={'units': 'm^2 s^-2', 'long_name': 'Barotropic Flux Vector (Phi component)'}
    )
    
    # === Add global attributes ===
    ds.attrs['title'] = f'LWA and Reference State Analysis Output ({qgfield_type})'
    ds.attrs['institution'] = 'Generated by analyze_nc_dataset.py'
    ds.attrs['source'] = 'FALWA package'
    ds.attrs['falwa_version'] = falwa.__version__
    ds.attrs['qgfield_type'] = qgfield_type
    ds.attrs['history'] = f'Created on {dt.datetime.now().isoformat()}'
    ds.attrs['references'] = ('Neal, E., Huang, C. S., & Nakamura, N. (2022). '
                              'The 2021 Pacific Northwest heat wave and associated blocking: '
                              'meteorology and the role of an upstream cyclone as a diabatic source of wave activity. '
                              'Geophysical Research Letters, 49(8), e2021GL097699.')
    ds.attrs['eq_boundary_index'] = eq_boundary_index
    ds.attrs['equator_idx'] = qgfield.equator_idx
    ds.attrs['analysis_timestamp'] = str(timestamp)
    
    # Generate output filename
    filepath = os.path.join(output_dir, filename)
    
    # Save to netCDF
    ds.to_netcdf(filepath)
    print(f"    Saved: {filename}")
    
    return filepath


def plot_3d_variables(qgfield, xlon, ylat, plev_idx, timestamp, output_dir, dpi=150):
    """
    Plot 3D variables at a selected pressure level.
    
    Parameters
    ----------
    qgfield : QGFieldNHN22 or QGFieldNH18
        Object containing computed quantities
    xlon, ylat : array-like
        Coordinate arrays
    plev_idx : int
        Pressure level index to plot
    timestamp : datetime
        Timestamp for the title
    output_dir : str
        Output directory for plots
    dpi : int
        DPI for saved figures
    """
    variables_3d = [
        (qgfield.qgpv, 'QGPV', 'Quasigeostrophic Potential Vorticity'),
        (qgfield.lwa, 'LWA', 'Local Wave Activity'),
        (qgfield.interpolated_u, 'U_interp', 'Interpolated Zonal Wind'),
        (qgfield.interpolated_v, 'V_interp', 'Interpolated Meridional Wind'),
    ]
    
    print(f"  Plotting 3D variables at pressure level index {plev_idx}...")
    
    for variable, short_name, long_name in variables_3d:
        fig, ax = plt.subplots(figsize=(12, 6))
        cf = ax.contourf(xlon, ylat[1:-1], variable[plev_idx, 1:-1, :], 50, cmap='jet')
        
        # Add white band at the equator for LWA
        if short_name == 'LWA':
            ax.axhline(y=0, c='w', lw=30)
        
        plt.colorbar(cf, ax=ax)
        ax.set_ylabel('Latitude (deg)')
        ax.set_xlabel('Longitude (deg)')
        ax.set_title(f'{long_name} at 240hPa | {timestamp}')
        
        filename = f'{short_name}_plev{plev_idx}_{timestamp.strftime("%Y%m%d_%H%M")}.png'
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {filename}")


def plot_reference_states(qgfield, ylat, height, eq_boundary_index, timestamp, output_dir, dpi=150):
    """
    Plot reference states on y-z plane.
    
    Parameters
    ----------
    qgfield : QGFieldNHN22 or QGFieldNH18
        Object containing computed quantities
    ylat : array-like
        Latitude array
    height : array-like
        Pseudoheight array
    eq_boundary_index : int
        Equatorward boundary index
    timestamp : datetime
        Timestamp for the title
    output_dir : str
        Output directory for plots
    dpi : int
        DPI for saved figures
    """
    variables_yz = [
        (qgfield.qref, 'Qref', 'Reference QGPV'),
        (qgfield.uref, 'Uref', 'Reference Zonal Wind'),
        (qgfield.ptref, 'PTref', 'Reference Potential Temperature'),
    ]
    
    print("  Plotting reference states...")
    
    equator_idx = qgfield.equator_idx
    
    for variable, short_name, long_name in variables_yz:
        # Mask out equatorward region outside analysis boundary
        mask = np.zeros_like(variable)
        mask[:, equator_idx - eq_boundary_index - 1:equator_idx + eq_boundary_index + 1] = np.nan
        variable_masked = np.ma.array(variable, mask=mask)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        cf = ax.contourf(ylat[1:-1], height[1:-1], variable_masked[1:-1, 1:-1], 50, cmap='jet')
        ax.axvline(x=0, c='w', lw=30)
        ax.set_xlabel('Latitude (deg)')
        ax.set_ylabel('Pseudoheight (m)')
        plt.colorbar(cf, ax=ax)
        ax.set_title(f'{long_name} | {timestamp}')
        
        filename = f'{short_name}_{timestamp.strftime("%Y%m%d_%H%M")}.png'
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {filename}")


def plot_barotropic_quantities(qgfield, xlon, ylat, timestamp, output_dir, dpi=150):
    """
    Plot vertically averaged (barotropic) quantities on x-y plane.
    
    Parameters
    ----------
    qgfield : QGFieldNHN22 or QGFieldNH18
        Object containing computed quantities
    xlon, ylat : array-like
        Coordinate arrays
    timestamp : datetime
        Timestamp for the title
    output_dir : str
        Output directory for plots
    dpi : int
        DPI for saved figures
    """
    variables_xy = [
        (qgfield.u_baro, 'U_baro', 'Barotropic Zonal Wind'),
        (qgfield.lwa_baro, 'LWA_baro', 'Barotropic LWA'),
        (qgfield.adv_flux_f1, 'F1', 'Advective Flux F1'),
        (qgfield.adv_flux_f2, 'F2', 'Advective Flux F2'),
        (qgfield.adv_flux_f3, 'F3', 'Advective Flux F3'),
        (qgfield.convergence_zonal_advective_flux, 'Flux_conv', 'Advective Flux Convergence'),
        (qgfield.divergence_eddy_momentum_flux, 'Eddy_mom_div', 'Eddy Momentum Flux Divergence'),
        (qgfield.meridional_heat_flux, 'Merid_heat', 'Meridional Heat Flux'),
        (qgfield.flux_vector_lambda_baro, 'Flux_lambda', 'Flux Vector Lambda (Barotropic)'),
        (qgfield.flux_vector_phi_baro, 'Flux_phi', 'Flux Vector Phi (Barotropic)'),
    ]
    
    print("  Plotting barotropic quantities...")
    
    for variable, short_name, long_name in variables_xy:
        fig, ax = plt.subplots(figsize=(12, 6))
        cf = ax.contourf(xlon, ylat[1:-1], variable[1:-1, :], 50, cmap='jet')
        ax.axhline(y=0, c='w', lw=30)
        ax.set_ylabel('Latitude (deg)')
        ax.set_xlabel('Longitude (deg)')
        plt.colorbar(cf, ax=ax)
        ax.set_title(f'{long_name} | {timestamp}')
        
        filename = f'{short_name}_{timestamp.strftime("%Y%m%d_%H%M")}.png'
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {filename}")


def create_summary_plot(qgfield, xlon, ylat, height, plev_idx, eq_boundary_index, 
                        timestamp, output_dir, dpi=150):
    """
    Create a summary figure with key variables.
    
    Parameters
    ----------
    qgfield : QGFieldNHN22 or QGFieldNH18
        Object containing computed quantities
    xlon, ylat : array-like
        Coordinate arrays
    height : array-like
        Pseudoheight array
    plev_idx : int
        Pressure level index
    eq_boundary_index : int
        Equatorward boundary index
    timestamp : datetime
        Timestamp for the title
    output_dir : str
        Output directory for plots
    dpi : int
        DPI for saved figures
    """
    print("  Creating summary plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. QGPV at selected level
    ax = axes[0, 0]
    cf = ax.contourf(xlon, ylat[1:-1], qgfield.qgpv[plev_idx, 1:-1, :], 50, cmap='jet')
    ax.axhline(y=0, c='w', lw=10)
    plt.colorbar(cf, ax=ax)
    ax.set_ylabel('Latitude (deg)')
    ax.set_xlabel('Longitude (deg)')
    ax.set_title('QGPV at 240hPa')
    
    # 2. LWA at selected level
    ax = axes[0, 1]
    cf = ax.contourf(xlon, ylat[1:-1], qgfield.lwa[plev_idx, 1:-1, :], 50, cmap='jet')
    ax.axhline(y=0, c='w', lw=10)
    plt.colorbar(cf, ax=ax)
    ax.set_ylabel('Latitude (deg)')
    ax.set_xlabel('Longitude (deg)')
    ax.set_title('LWA at 240hPa')
    
    # 3. Barotropic LWA
    ax = axes[0, 2]
    cf = ax.contourf(xlon, ylat[1:-1], qgfield.lwa_baro[1:-1, :], 50, cmap='jet')
    ax.axhline(y=0, c='w', lw=10)
    plt.colorbar(cf, ax=ax)
    ax.set_ylabel('Latitude (deg)')
    ax.set_xlabel('Longitude (deg)')
    ax.set_title('Barotropic LWA')
    
    # 4. Qref
    ax = axes[1, 0]
    equator_idx = qgfield.equator_idx
    mask = np.zeros_like(qgfield.qref)
    mask[:, equator_idx - eq_boundary_index - 1:equator_idx + eq_boundary_index + 1] = np.nan
    qref_masked = np.ma.array(qgfield.qref, mask=mask)
    cf = ax.contourf(ylat[1:-1], height[1:-1], qref_masked[1:-1, 1:-1], 50, cmap='jet')
    ax.axvline(x=0, c='w', lw=10)
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel('Latitude (deg)')
    ax.set_ylabel('Pseudoheight (m)')
    ax.set_title('Reference QGPV (Qref)')
    
    # 5. Uref
    ax = axes[1, 1]
    mask = np.zeros_like(qgfield.uref)
    mask[:, equator_idx - eq_boundary_index - 1:equator_idx + eq_boundary_index + 1] = np.nan
    uref_masked = np.ma.array(qgfield.uref, mask=mask)
    cf = ax.contourf(ylat[1:-1], height[1:-1], uref_masked[1:-1, 1:-1], 50, cmap='jet')
    ax.axvline(x=0, c='w', lw=10)
    plt.colorbar(cf, ax=ax)
    ax.set_xlabel('Latitude (deg)')
    ax.set_ylabel('Pseudoheight (m)')
    ax.set_title('Reference Zonal Wind (Uref)')
    
    # 6. Flux convergence
    ax = axes[1, 2]
    cf = ax.contourf(xlon, ylat[1:-1], qgfield.convergence_zonal_advective_flux[1:-1, :], 50, cmap='jet')
    ax.axhline(y=0, c='w', lw=10)
    plt.colorbar(cf, ax=ax)
    ax.set_ylabel('Latitude (deg)')
    ax.set_xlabel('Longitude (deg)')
    ax.set_title('Advective Flux Convergence')
    
    fig.suptitle(f'LWA Analysis Summary | {timestamp}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'summary_{timestamp.strftime("%Y%m%d_%H%M")}.png'
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {filename}")


def main():
    """Main function to run the analysis."""
    args = parse_arguments()
    
    print("=" * 60)
    print(f"LWA and Reference State Analysis ({args.qgfield_type})")
    print(f"FALWA version: {falwa.__version__}")
    print(f"Run date: {dt.date.today()}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    u_data, v_data, t_data, time_array, ntimes = load_data(
        args.u_file, args.v_file, args.t_file
    )
    
    # Setup grid
    xlon, ylat, plev, height, kmax, dz, hh = setup_grid(u_data)
    
    # Calculate timestamp
    try:
        base_time = time_array[args.tstep].values
        timestamp = np.datetime64(base_time, 's').astype(dt.datetime)
    except:
        # Fallback if time parsing fails
        timestamp = dt.datetime(2005, 1, 23, 0, 0) + dt.timedelta(hours=6 * args.tstep)
    
    print(f"\nAnalyzing time step {args.tstep}/{ntimes}: {timestamp}")
    
    # Extract fields
    print("\nExtracting fields...")
    uu, vv, tt = extract_fields(u_data, v_data, t_data, args.tstep)
    
    # Compute LWA and fluxes
    print("\nComputing LWA and reference states...")
    qgfield = compute_lwa_and_fluxes(
        xlon, ylat, plev, uu, vv, tt,
        eq_boundary_index=args.eq_boundary_index,
        qgfield_type=args.qgfield_type
    )
    
    # Save to netCDF (unless --no_netcdf flag is set)
    if not args.no_netcdf:
        output_nc_filename = args.output_nc_filename
        print("\nSaving to netCDF...")
        nc_filepath = save_to_netcdf(
            qgfield, xlon, ylat, height, timestamp,
            str(output_dir), args.eq_boundary_index, output_nc_filename,
            qgfield_type=args.qgfield_type
        )
    
    # Generate plots (unless --no_plots flag is set)
    if not args.no_plots:
        print("\nGenerating plots...")
        
        plot_3d_variables(
            qgfield, xlon, ylat, args.plev_idx, timestamp, 
            str(output_dir), dpi=args.dpi
        )
        
        plot_reference_states(
            qgfield, ylat, height, args.eq_boundary_index, timestamp,
            str(output_dir), dpi=args.dpi
        )
        
        plot_barotropic_quantities(
            qgfield, xlon, ylat, timestamp,
            str(output_dir), dpi=args.dpi
        )
        
        create_summary_plot(
            qgfield, xlon, ylat, height, args.plev_idx, args.eq_boundary_index,
            timestamp, str(output_dir), dpi=args.dpi
        )
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)
    
    # Close datasets
    u_data.close()
    v_data.close()
    t_data.close()


if __name__ == "__main__":
    main()

