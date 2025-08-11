"""
------------------------------------------
File name: utilities.py
Author: Clare Huang
"""
import numpy as np
from math import pi, exp
from typing import Optional, Tuple


def static_stability(height: np.ndarray, area: np.ndarray, theta: np.ndarray,
                     s_et: Optional[int] = None, n_et: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the vertical gradient of hemispheric-averaged potential temperature.

    This computes d(theta)/dz, i.e. the static stability term in the definition
    of QGPV in eq.(3) of Huang and Nakamura (2016), by central differencing.
    At the boundary, the static stability is estimated by forward/backward
    differencing involving two adjacent z-grid points:

        i.e. stat_n[0] = (t0_n[1] - t0_n[0]) / (height[1] - height[0])
            stat_n[-1] = (t0_n[-2] - t0_n[-1]) / (height[-2] - height[-1])

    Please make inquiries and report issues via Github:
    https://github.com/csyhuang/hn2016_falwa/issues

    Parameters
    ----------
    height : np.ndarray
        Array of z-coordinate [in meters] with dimension (kmax,), equally spaced.
    area : np.ndarray
        Two-dimension numpy array specifying differential areal element of each
        grid point; dimension = (nlat, nlon).
    theta : np.ndarray
        Matrix of potential temperature [K] with dimension (kmax, nlat, nlon)
        or (kmax, nlat).
    s_et : int, optional
        Index of the latitude that defines the boundary of the Southern
        hemispheric domain. Defaults to nlat/2.
    n_et : int, optional
        Index of the latitude that defines the boundary of the Northern
        hemispheric domain. Defaults to nlat/2.

    Returns
    -------
    t0_n : np.ndarray
        Area-weighted average of potential temperature in the Northern
        hemispheric domain with dimension (kmax,).
    t0_s : np.ndarray
        Area-weighted average of potential temperature in the Southern
        hemispheric domain with dimension (kmax,).
    stat_n : np.ndarray
        Static stability in the Northern hemispheric domain with dimension (kmax,).
    stat_s : np.ndarray
        Static stability in the Southern hemispheric domain with dimension (kmax,).
    """

    nlat = theta.shape[1]
    if s_et is None:
        s_et = nlat // 2
    if n_et is None:
        n_et = nlat // 2

    stat_n = np.zeros(theta.shape[0])
    stat_s = np.zeros(theta.shape[0])

    if theta.ndim == 3:
        zonal_mean = np.mean(theta, axis=-1)
    elif theta.ndim == 2:
        zonal_mean = theta

    if area.ndim == 2:
        area_zonal_mean = np.mean(area, axis=-1)
    elif area.ndim == 1:
        area_zonal_mean = area

    csm_n_et = np.sum(area_zonal_mean[-n_et:])
    csm_s_et = np.sum(area_zonal_mean[:s_et])

    t0_n = np.sum(zonal_mean[:, -n_et:] * area_zonal_mean[np.newaxis, -n_et:],
                  axis=-1) / csm_n_et
    t0_s = np.sum(zonal_mean[:, :s_et] * area_zonal_mean[np.newaxis, :s_et],
                  axis=-1) / csm_s_et

    stat_n[1:-1] = (t0_n[2:] - t0_n[:-2]) / (height[2:] - height[:-2])
    stat_s[1:-1] = (t0_s[2:] - t0_s[:-2]) / (height[2:] - height[:-2])
    stat_n[0] = (t0_n[1] - t0_n[0]) / (height[1] - height[0])
    stat_n[-1] = (t0_n[-2] - t0_n[-1]) / (height[-2] - height[-1])
    stat_s[0] = (t0_s[1] - t0_s[0]) / (height[1] - height[0])
    stat_s[-1] = (t0_s[-2] - t0_s[-1]) / (height[-2] - height[-1])

    return t0_n, t0_s, stat_n, stat_s


def compute_qgpv_givenvort(
        omega: float, nlat: int, nlon: int, kmax: int, unih: np.ndarray, ylat: np.ndarray,
        avort: np.ndarray, potential_temp: np.ndarray, t0_cn: np.ndarray, t0_cs: np.ndarray,
        stat_cn: np.ndarray, stat_cs: np.ndarray, nlat_s: Optional[int] = None,
        scale_height: float = 7000.) -> Tuple[np.ndarray, np.ndarray]:
    """Compute QGPV from absolute vorticity, potential temperature and static stability.

    The computation is done for a hemispheric domain.

    Please make inquiries and report issues via Github:
    https://github.com/csyhuang/hn2016_falwa/issues

    Parameters
    ----------
    omega : float
        Rotation rate of the planet.
    nlat : int
        Latitudinal dimension of the latitude grid.
    nlon : int
        Longitudinal dimension of the longitude grid.
    kmax : int
        Vertical dimension of the height grid.
    unih : np.ndarray
        Numpy array of height in [meters]; dimension = (kmax,).
    ylat : np.ndarray
        Numpy array of latitudes in [degrees]; dimension = (nlat,).
    avort : np.ndarray
        3D numpy array of absolute vorticity (relative vorticity + planetary
        vorticity) in [1/s]; dimension = (kmax, nlat, nlon).
    potential_temp : np.ndarray
        3D numpy array of potential temperature in [K];
        dimension = (kmax, nlat, nlon).
    t0_cn : np.ndarray
        Area-weighted average of potential temperature in the Northern
        hemispheric domain with dimension (kmax,).
    t0_cs : np.ndarray
        Area-weighted average of potential temperature in the Southern
        hemispheric domain with dimension (kmax,).
    stat_cn : np.ndarray
        Static stability in the Northern hemispheric domain with dimension (kmax,).
    stat_cs : np.ndarray
        Static stability in the Southern hemispheric domain with dimension (kmax,).
    nlat_s : int, optional
        Latitudinal index for the poleward boundary of the hemisphere.
        Defaults to nlat // 2.
    scale_height : float, optional
        Scale height of the atmosphere in [m]. Default is 7000.

    Returns
    -------
    QGPV : np.ndarray
        3D numpy array of quasi-geostrophic potential vorticity;
        dimension = (kmax, nlat, nlon).
    dzdiv : np.ndarray
        3D numpy array of the stretching term in QGPV;
        dimension = (kmax, nlat, nlon).
    """

    if nlat_s is None:
        nlat_s = nlat // 2

    # --- Next, calculate PV ---
    qgpv = np.empty_like(potential_temp)  # av1+av2+av3+dzdiv

    av1 = np.ones((kmax, nlat, nlon)) * \
        2*omega*np.sin(np.deg2rad(ylat[np.newaxis, :, np.newaxis]))

    # Calculate the z-divergence term
    zdiv = np.empty_like(potential_temp)
    dzdiv = np.empty_like(potential_temp)
    for kk in range(kmax):  # This is more efficient
        zdiv[kk, :nlat_s, :] = exp(-unih[kk]/scale_height)*(
            potential_temp[kk, :nlat_s, :] - t0_cs[kk]
        )/stat_cs[kk]
        zdiv[kk, -nlat_s:, :] = exp(-unih[kk]/scale_height)*(
            potential_temp[kk, -nlat_s:, :] - t0_cn[kk]
        )/stat_cn[kk]

        dzdiv[1:kmax-1, :, :] = np.exp(
            unih[1:kmax-1, np.newaxis, np.newaxis]/scale_height
        ) * (zdiv[2:kmax, :, :]-zdiv[0:kmax-2, :, :]) \
            / (unih[2:kmax, np.newaxis, np.newaxis] -
               unih[0:kmax-2, np.newaxis, np.newaxis])

        dzdiv[0, :, :] = exp(unih[0]/scale_height) *\
            (zdiv[1, :, :]-zdiv[0, :, :]) /\
            (unih[1, np.newaxis, np.newaxis] - unih[0, np.newaxis, np.newaxis])
        dzdiv[kmax-1, :, :] = exp(unih[kmax-1]/scale_height) *\
            (zdiv[kmax-1, :, :]-zdiv[kmax-2, :, :]) /\
            (unih[kmax-1, np.newaxis, np.newaxis] -
             unih[kmax-2, np.newaxis, np.newaxis])

        qgpv = avort + dzdiv * av1
    return qgpv, dzdiv


def zonal_convergence(field: np.ndarray, clat: np.ndarray, dlambda: float,
                      planet_radius: float = 6.378e+6, tol: float = 1.e-5) -> np.ndarray:
    """Compute the zonal component of the convergence operator of a field.

    This computes on the spherical surface:
    -1/(planet_radius * cos(lat)) * partial d(f(lat, lon))/partial d(lon)

    Please make inquiries and report issues via Github:
    https://github.com/csyhuang/hn2016_falwa/issues

    Parameters
    ----------
    field : np.ndarray
        An arbitrary field to compute zonal divergence on, with
        dimension (nlat, nlon).
    clat : np.ndarray
        Numpy array of cosine latitude; dimension (nlat,).
    dlambda : float
        Differential element of longitude.
    planet_radius : float, optional
        Radius of the planet in meters. Default is 6.378e+6 (Earth's radius).
    tol : float, optional
        Tolerance below which clat is considered zero. Grid points with
        clat below tolerance will have a returned result of 0 to avoid
        division by zero. Default is 1.e-5.

    Returns
    -------
    np.ndarray
        Zonal convergence of field with the same dimension as field,
        i.e., (nlat, nlon).
    """

    zonal_diff = np.zeros_like(field)
    zonal_diff[:, 1:-1] = field[:, 2:] - field[:, :-2]
    zonal_diff[:, 0] = field[:, 1] - field[:, -1]
    zonal_diff[:, -1] = field[:, 0] - field[:, -2]

    # This is to avoid divided by zero
    finite_clat = np.abs(clat) > tol

    zonal_diff[finite_clat, :] = zonal_diff[finite_clat, :] * \
                                 (-1./(planet_radius * clat[finite_clat, np.newaxis] * 2. * dlambda))

    return zonal_diff


def curl_2d(ufield: np.ndarray, vfield: np.ndarray, clat: np.ndarray, dlambda: float,
            dphi: float, planet_radius: float = 6.378e+6) -> np.ndarray:
    """Compute the curl of velocity on a pressure level in spherical coordinates.

    Assumes regular latitude and longitude [in degree] grid.

    Parameters
    ----------
    ufield : np.ndarray
        2D array of zonal wind.
    vfield : np.ndarray
        2D array of meridional wind.
    clat : np.ndarray
        1D array of cosine of latitude.
    dlambda : float
        Grid spacing in longitude in radians.
    dphi : float
        Grid spacing in latitude in radians.
    planet_radius : float, optional
        Radius of the planet. Default is Earth's radius (6.378e+6 m).

    Returns
    -------
    np.ndarray
        2D array of the vertical component of the curl.
    """

    ans = np.zeros_like(ufield)
    ans[1:-1, 1:-1] = (vfield[1:-1, 2:] - vfield[1:-1, :-2])/(2.*dlambda) - \
                      (ufield[2:, 1:-1] * clat[2:, np.newaxis] -
                       ufield[:-2, 1:-1] * clat[:-2, np.newaxis])/(2.*dphi)
    ans[0, :] = 0.0
    ans[-1, :] = 0.0
    ans[1:-1, 0] = ((vfield[1:-1, 1] - vfield[1:-1, -1]) / (2. * dlambda) -
                    (ufield[2:, 0] * clat[2:] -
                     ufield[:-2, 0] * clat[:-2]) / (2. * dphi))
    ans[1:-1, -1] = ((vfield[1:-1, 0] - vfield[1:-1, -2]) / (2. * dlambda) -
                     (ufield[2:, -1] * clat[2:] -
                      ufield[:-2, -1] * clat[:-2]) / (2. * dphi))
    ans[1:-1, :] = ans[1:-1, :] / planet_radius / clat[1:-1, np.newaxis]
    return ans


def z_derivative_of_prod(
    stat_n: np.ndarray, stat_s: np.ndarray, kmax: int, equator_idx: int, dz: float,
    density_decay: np.ndarray, gfunc: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    """Compute the z-derivative of a product involving static stability.

    This computes the expression:

        f * exp(z/H) * d/dz [ exp(-z/H) / static_stability * g(z, phi, lambda) ]

    Parameters
    ----------
    stat_n : np.ndarray
        Static stability per pressure level in Northern Hemisphere. Dim (kmax,).
    stat_s : np.ndarray
        Static stability per pressure level in Southern Hemisphere. Dim (kmax,).
    kmax : int
        Number of pseudoheight levels.
    equator_idx : int
        Latitudinal index of the equator (phi = 0).
    dz : float
        Differential element of pseudoheight.
    density_decay : np.ndarray
        exp(-z/H). Dim (kmax,).
    gfunc : np.ndarray
        g(z, phi, lambda). Dim (kmax, nlat, nlon).
    multiplier : np.ndarray
        f * exp(z/H), where f is Coriolis parameter. Dim (kmax, nlat).

    Returns
    -------
    np.ndarray
        Array of dim (kmax, nlat, nlon) that is the result.
    """
    # Make static_stability_temp (kmax, nlat) ndarray
    static_stability_temp = np.concatenate([
        np.ones((kmax, equator_idx - 1)) * stat_s[:, np.newaxis],
        np.ones((kmax, 1)) * 0.5 * (stat_s+stat_n)[:, np.newaxis],
        np.ones((kmax, equator_idx - 1)) * stat_n[:, np.newaxis]], axis=1)

    # the term to be differentiated in z is of dim (kmax, nlat, nlon)
    to_be_diff = density_decay[:, np.newaxis, np.newaxis] * gfunc / static_stability_temp[:, :, np.newaxis]
    z_der = np.zeros_like(to_be_diff)
    z_der[1:-1, :, :] = (to_be_diff[2:, :, :] - to_be_diff[:-2, :, :])/(2 * dz) * multiplier[1:-1, :, np.newaxis]

    # Estimate boundary term with no-so-good approximation...
    z_der[0, :, :] = multiplier[0, :, np.newaxis] * (to_be_diff[1, :, :] - to_be_diff[0, :, :]) / dz
    z_der[-1, :, :] = multiplier[-1, :, np.newaxis] * (to_be_diff[-1, :, :] - to_be_diff[-2, :, :]) / dz

    return z_der
