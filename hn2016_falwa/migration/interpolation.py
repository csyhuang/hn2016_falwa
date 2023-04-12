"""
-------------------------------------------------------------------------------------------------------------------
File name: interpolation.py
Author: Clare Huang
Created on: 2023/4/10
Description: This module contains functions that replace the f2py module "interpolate_fields".
  The helper functions are:
  - interpolate_onto_uniform_pseudoheight_grid
  - compute_static_stability
  - compute_pv_from_stat_t0
  - calculate_absolute_vorticity
-------------------------------------------------------------------------------------------------------------------
"""
from typing import Tuple, Optional
import numpy as np
from scipy.interpolate import interp1d
from hn2016_falwa.constant import P_GROUND, SCALE_HEIGHT, CP, DRY_GAS_CONSTANT, EARTH_RADIUS, EARTH_OMEGA


def interpolate_fields(
    nlon: int, nlat: int, nlev: int, kmax: int, equator_idx: int,
    uu: np.array, vv: np.array, temp: np.array, plev: np.array, height: np.array,
    zlev: np.array, ylat: np.array, clat: np.array, slat: np.array,
    dz: float, dphi: float,
    jd: Optional[int] = None, use_global_stat_t0: bool = False, smooth_stretch_term: bool = False,
    stat_boundary_treatment: str = 'nh18',
    hh: float = SCALE_HEIGHT, aa: float = EARTH_RADIUS, omega: float = EARTH_OMEGA,
    rkappa: float = DRY_GAS_CONSTANT/CP) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    This function replaces two fortran modules:
        - interpolate_fields.f90::interpolate_fields
        - interpolate_fields_dirinv.f90::interpolate_fields_direct_inv

    It handles 3 scenario (different domains + BC):
        - NS11: jd is None; equator_idx = nlat/2 + 1; smooth_stretch_term = True
        - NH18: jd = 0; equator_idx = nlat/2 + 1 (or 0 if hemispheric data is provided); smooth_stretch_term = True
        - NHN22: jd = 5; equator_idx = nlat/2 + 1 (or 0 if hemispheric data is provided); smooth_stretch_term = False
    """
    # *** The variables below shall be input from the QGField manager ***
    # ~ Physical constants ~
    # rkappa: float = rr/cp
    # dphi: float = pi/float(nlat-1)
    # ~ Coordinate-related arrays ~
    # zlev: np.array = -hh * np.log(plev/P_GROUND)  # TODO: move this to QGField and make this an input of this function
    # ylat: np.array = np.linspace(-90, 90, nlat)
    # clat: np.array = np.cos(np.deg2rad(ylat))  # (nlat,)
    # slat: np.array = np.sin(np.deg2rad(ylat))  # (nlat,)

    # *** Set logic at the beginning ***
    hemispheric_grid: bool = (equator_idx == 0)  # Whether input data contains global (False) or hemispheric data (True)
    hemispheric_domain: bool = (jd is not None)  # Whether LWA computation shall be done in global/hemispheric domain
    if hemispheric_domain:
        lat_extend: Optional[int] = (nlat - jd) if hemispheric_grid else (equator_idx - jd)
    else:
        lat_extend = None

    # *** Initialize arrays ***
    stat = np.zeros(kmax)
    stat_n = np.zeros(kmax)
    stat_s = np.zeros(kmax)
    pv = np.zeros((kmax, nlat, nlon))

    ut, vt, theta = interpolate_onto_uniform_pseudoheight_grid(plev, zlev, height, uu, vv, temp, rkappa)

    # *** Absolute vorticity ***
    avort = calculate_absolute_vorticity(kmax, nlat, nlon, omega, slat, ut, vt, aa, clat, dphi)

    # *** Reference theta (global/hemispheric) and Static stability computed w/ central differencing ***
    if hemispheric_domain and (not hemispheric_grid):
        print("Debug: route 1.1")
        print(f"equator_idx = {equator_idx}")
        ts0, stat_s = compute_static_stability(
            theta[:, :equator_idx, :], clat[:equator_idx], stat_s, dz, boundary_treatment=stat_boundary_treatment)
        tn0, stat_n = compute_static_stability(
            theta[:, -equator_idx:, :], clat[-equator_idx:], stat_n, dz, boundary_treatment=stat_boundary_treatment)
        if use_global_stat_t0:
            print("Debug: route 2.1")
            # This branch of logic would recover NH18 results
            t0 = (ts0 + tn0)/2.
            stat = (stat_s + stat_n) / 2.
            pv[:, :equator_idx, :] = -compute_pv_from_stat_t0(
                theta[:, :equator_idx, :][:, ::-1, :], -avort[:, :equator_idx, :][:, ::-1, :], height, hh, t0, stat, omega,
                -slat[:equator_idx][::-1], smooth_stretch_term=smooth_stretch_term)[:, ::-1, :]  # this is flipped
            pv[:, -equator_idx:, :] = compute_pv_from_stat_t0(
                theta[:, -equator_idx:, :], avort[:, -equator_idx:, :], height, hh, t0, stat, omega,
                slat[-equator_idx:], smooth_stretch_term=smooth_stretch_term)
            return pv, ut, vt, avort, theta, stat, t0
        else:
            print("Debug: route 2.2")
            # *** Hemispheric domain instead ***
            pv[:, :equator_idx, :] = -compute_pv_from_stat_t0(
                theta[:, :equator_idx, :][:, ::-1, :], -avort[:, :equator_idx, :][:, ::-1, :], height, hh, ts0, stat_s, omega,
                -slat[:equator_idx][::-1], smooth_stretch_term=smooth_stretch_term)[:, ::-1, :]  # this is flipped
            pv[:, -equator_idx:, :] = compute_pv_from_stat_t0(
                theta[:, -equator_idx:, :], avort[:, -equator_idx:, :], height, hh, tn0, stat_n, omega,
                slat[-equator_idx:], smooth_stretch_term=smooth_stretch_term)
            return pv, ut, vt, avort, theta, (stat_s, stat_n), (ts0, tn0)
    else:  # Return global static stability
        print("Debug: route 1.2")
        t0, stat = compute_static_stability(theta, clat, stat, dz)
        pv = compute_pv_from_stat_t0(
            theta, avort, height, hh, t0, stat, omega, slat, smooth_stretch_term=smooth_stretch_term)
        return pv, ut, vt, avort, theta, stat, t0


def interpolate_onto_uniform_pseudoheight_grid(plev, zlev, height, uu, vv, temp, rkappa):
    """
    Interpolate physical fields from zlev grid to height grid. Applicable to both global and hemispheric data.
    """
    # Get potential temperature
    # plev and zlev are of dim (nlev)
    theta_uninterp: np.array = temp * ((P_GROUND / plev[:, np.newaxis, np.newaxis]) ** rkappa)  # (nlev, nlat, nlon)

    # Interpolate uu, vv, theta_uninterp (nlev, nlat, nlon) to ut, vt, theta (kmax, nlat, nlon)
    interp_u = interp1d(zlev, uu, axis=0)
    interp_v = interp1d(zlev, vv, axis=0)
    interp_t = interp1d(zlev, theta_uninterp, axis=0)
    ut = interp_u(height)
    vt = interp_v(height)
    theta = interp_t(height)
    return ut, vt, theta


def calculate_absolute_vorticity(kmax, nlat, nlon, omega, slat, ut, vt, aa, clat, dphi):
    avort = np.zeros((kmax, nlat, nlon))
    # *** Absolute vorticity (Interior) ***
    avort[:, 1:-1, 1:-1] = \
        2. * omega * slat[np.newaxis, 1:-1, np.newaxis] \
        + (vt[:, 1:-1, 2:] - vt[:, 1:-1, :-2])/(2. * aa * clat[np.newaxis, 1:-1, np.newaxis] * dphi) \
        - (ut[:, 2:, 1:-1] * clat[np.newaxis, 2:, np.newaxis] - ut[:, :-2, 1:-1] * clat[np.newaxis, :-2, np.newaxis]) / (2. * aa * clat[np.newaxis, 1:-1, np.newaxis] * dphi)
    # *** Absolute vorticity (Longitude boundary) ***
    avort[:, 1:-1, 0] = \
        2. * omega * slat[np.newaxis, 1:-1] \
        + (vt[:, 1:-1, 1] - vt[:, 1:-1, -1])/(2. * aa * clat[np.newaxis, 1:-1] * dphi) \
        - (ut[:, 2:, 0] * clat[2:] - ut[:, :-2, 0] * clat[np.newaxis, :-2]) / (2. * aa * clat[np.newaxis, 1:-1] * dphi)
    avort[:, 1:-1, -1] = \
        2. * omega * slat[np.newaxis, 1:-1] \
        + (vt[:, 1:-1, 0] - vt[:, 1:-1, -2])/(2. * aa * clat[np.newaxis, 1:-1] * dphi) \
        - (ut[:, 2:, -1] * clat[2:] - ut[:, :-2, -1] * clat[np.newaxis, :-2]) / (2. * aa * clat[np.newaxis, 1:-1] * dphi)
    # *** Absolute vorticity (Poles) ***
    avort[:, 0, :] = avort[:, 1, :].mean(axis=1)[:, np.newaxis]
    avort[:, -1, :] = avort[:, -2, :].mean(axis=1)[:, np.newaxis]
    return avort


def compute_static_stability(theta: np.array, clat: np.array, stat: np.array, dz: float, boundary_treatment: str = 'nh18'):
    print(f"In compute_static_stability. Boundary treatment: {boundary_treatment}. dz: {dz}.")
    t0 = (theta.mean(axis=-1) * clat[np.newaxis, :]).sum(axis=1) / clat.sum()  # zonal mean theta (kmax,)
    # Interior values: this is correct
    stat[1:-1] = (t0[2:] - t0[:-2]) / (2. * dz)
    # Boundary values
    if boundary_treatment == 'nh18':
        stat[-1] = (t0[-1] - t0[-2]) / dz
        stat[0] = (t0[1] - t0[0]) / dz
    elif boundary_treatment == 'nhn22':
        stat[-1] = 2. * stat[-2] - stat[-3]
        stat[0] = 2. * stat[1] - stat[2]
    return t0, stat


def compute_pv_from_stat_t0(theta, avort, height, hh, t0, stat, omega, slat, smooth_stretch_term=True):
    pv = np.zeros_like(theta)
    altp = np.exp(-height[2:, np.newaxis, np.newaxis] / hh) * (theta[2:, :, :] - t0[2:, np.newaxis, np.newaxis]) / stat[2:, np.newaxis, np.newaxis]
    altm = np.exp(-height[:-2, np.newaxis, np.newaxis] / hh) * (theta[:-2, :, :] - t0[:-2, np.newaxis, np.newaxis]) / stat[:-2, np.newaxis, np.newaxis]
    print(f"smooth_stretch_term = {smooth_stretch_term}")
    if smooth_stretch_term:
        zmav = avort.mean(axis=2)  # zonal mean abs vort
        strc = (altp - altm) * zmav[1:-1, :, np.newaxis] / (height[2:, np.newaxis, np.newaxis] - height[:-2, np.newaxis, np.newaxis])
    else:
        strc = (altp - altm) * 2 * omega * slat[np.newaxis, :, np.newaxis] / (height[2:, np.newaxis, np.newaxis] - height[:-2, np.newaxis, np.newaxis])
    print(f"strc.shape = {strc.shape}")  # (47,61,240)
    pv[1:-1, :, :] = avort[1:-1, :, :] + np.exp(height[1:-1, np.newaxis, np.newaxis] / hh) * strc
    return pv
