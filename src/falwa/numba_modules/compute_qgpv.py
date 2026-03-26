"""
Numba implementation of compute_qgpv.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `compute_qgpv` for computing quasi-geostrophic potential vorticity
and absolute vorticity.

.. versionadded:: 2.4.0

Notes
-----
This implementation is equivalent to the Fortran subroutine in
`f90_modules/compute_qgpv.f90` and produces numerically identical results.
The function is JIT-compiled using Numba for performance comparable to
the original Fortran implementation.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(cache=True)
def _compute_absolute_vorticity(
    ut: np.ndarray,
    vt: np.ndarray,
    nlon: int,
    nlat: int,
    kmax: int,
    aa: float,
    omega: float,
    dphi: float,
    pi: float
) -> np.ndarray:
    """
    Compute absolute vorticity from wind fields.
    
    Parameters
    ----------
    ut : np.ndarray
        Zonal wind field, shape (nlon, nlat, kmax)
    vt : np.ndarray
        Meridional wind field, shape (nlon, nlat, kmax)
    nlon : int
        Number of longitude points
    nlat : int
        Number of latitude points  
    kmax : int
        Number of vertical levels
    aa : float
        Earth radius
    omega : float
        Earth's rotation rate
    dphi : float
        Latitude grid spacing in radians
    pi : float
        Pi constant
        
    Returns
    -------
    avort : np.ndarray
        Absolute vorticity, shape (nlon, nlat, kmax)
    """
    avort = np.zeros((nlon, nlat, kmax), dtype=np.float64)
    
    for kk in range(kmax):
        # Interior latitudes (j=1 to nlat-2 in 0-indexed, corresponds to j=2 to nlat-1 in Fortran)
        for j in range(1, nlat - 1):
            # Fortran: phi0 = -90.+float(j-1)*180./float(nlat-1)
            # j in Fortran is 2 to nlat-1, so j-1 is 1 to nlat-2
            # In Python, j is 1 to nlat-2 (0-indexed), equivalent to Fortran j-1 = 1 to nlat-2
            phi0 = -90.0 + float(j) * 180.0 / float(nlat - 1)
            phi0 = phi0 * pi / 180.0
            phim = -90.0 + float(j - 1) * 180.0 / float(nlat - 1)
            phim = phim * pi / 180.0
            phip = -90.0 + float(j + 1) * 180.0 / float(nlat - 1)
            phip = phip * pi / 180.0
            
            cos_phi0 = np.cos(phi0)
            cos_phim = np.cos(phim)
            cos_phip = np.cos(phip)
            
            # Planetary vorticity
            av1 = 2.0 * omega * np.sin(phi0)
            
            # Interior longitudes (i=1 to nlon-2 in 0-indexed, corresponds to i=2 to nlon-1 in Fortran)
            for i in range(1, nlon - 1):
                # Relative vorticity components
                av2 = (vt[i + 1, j, kk] - vt[i - 1, j, kk]) / (2.0 * aa * cos_phi0 * dphi)
                av3 = -(ut[i, j + 1, kk] * cos_phip - ut[i, j - 1, kk] * cos_phim) / (2.0 * aa * cos_phi0 * dphi)
                avort[i, j, kk] = av1 + av2 + av3
            
            # Periodic boundary conditions for i=0 (Fortran i=1)
            av2 = (vt[1, j, kk] - vt[nlon - 1, j, kk]) / (2.0 * aa * cos_phi0 * dphi)
            av3 = -(ut[0, j + 1, kk] * cos_phip - ut[0, j - 1, kk] * cos_phim) / (2.0 * aa * cos_phi0 * dphi)
            avort[0, j, kk] = av1 + av2 + av3
            
            # Periodic boundary conditions for i=nlon-1 (Fortran i=nlon)
            av5 = (vt[0, j, kk] - vt[nlon - 2, j, kk]) / (2.0 * aa * cos_phi0 * dphi)
            av6 = -(ut[nlon - 1, j + 1, kk] * cos_phip - ut[nlon - 1, j - 1, kk] * cos_phim) / (2.0 * aa * cos_phi0 * dphi)
            avort[nlon - 1, j, kk] = av1 + av5 + av6
        
        # Pole values: average from neighboring latitude
        avs = 0.0
        avn = 0.0
        for i in range(nlon):
            avs += avort[i, 1, kk] / float(nlon)
            avn += avort[i, nlat - 2, kk] / float(nlon)
        
        # Set pole values (all longitudes get the same value)
        for i in range(nlon):
            avort[i, 0, kk] = avs       # South pole
            avort[i, nlat - 1, kk] = avn  # North pole
    
    return avort


@njit(cache=True)
def _compute_zonal_mean_vorticity(
    avort: np.ndarray,
    nlon: int,
    nlat: int,
    kmax: int
) -> np.ndarray:
    """
    Compute zonal mean of absolute vorticity.
    
    Parameters
    ----------
    avort : np.ndarray
        Absolute vorticity, shape (nlon, nlat, kmax)
    nlon : int
        Number of longitude points
    nlat : int
        Number of latitude points
    kmax : int
        Number of vertical levels
        
    Returns
    -------
    zmav : np.ndarray
        Zonal mean absolute vorticity, shape (nlat, kmax)
    """
    zmav = np.zeros((nlat, kmax), dtype=np.float64)
    
    for kk in range(kmax):
        for j in range(nlat):
            for i in range(nlon):
                zmav[j, kk] += avort[i, j, kk] / float(nlon)
    
    return zmav


@njit(cache=True)
def _compute_interior_pv(
    avort: np.ndarray,
    theta: np.ndarray,
    zmav: np.ndarray,
    height: np.ndarray,
    t0: np.ndarray,
    stat: np.ndarray,
    nlon: int,
    nlat: int,
    kmax: int,
    omega: float,
    hh: float,
    pi: float
) -> np.ndarray:
    """
    Compute interior potential vorticity with stretching term.
    
    Parameters
    ----------
    avort : np.ndarray
        Absolute vorticity, shape (nlon, nlat, kmax)
    theta : np.ndarray
        Potential temperature, shape (nlon, nlat, kmax)
    zmav : np.ndarray
        Zonal mean absolute vorticity, shape (nlat, kmax)
    height : np.ndarray
        Height levels, shape (kmax,)
    t0 : np.ndarray
        Reference temperature, shape (kmax,)
    stat : np.ndarray
        Static stability, shape (kmax,)
    nlon : int
        Number of longitude points
    nlat : int
        Number of latitude points
    kmax : int
        Number of vertical levels
    omega : float
        Earth's rotation rate
    hh : float
        Scale height
    pi : float
        Pi constant
        
    Returns
    -------
    pv : np.ndarray
        Potential vorticity, shape (nlon, nlat, kmax)
    """
    pv = np.zeros((nlon, nlat, kmax), dtype=np.float64)
    
    # Interior levels only (kk=1 to kmax-2 in 0-indexed, corresponds to kk=2 to kmax-1 in Fortran)
    for kk in range(1, kmax - 1):
        for j in range(nlat):
            phi0 = -90.0 + float(j) * 180.0 / float(nlat - 1)
            phi0 = phi0 * pi / 180.0
            # f = 2.0 * omega * np.sin(phi0)  # Not used in the actual calculation
            
            for i in range(nlon):
                # Stretching term calculation
                # altp corresponds to level kk+1, altm to level kk-1
                altp = np.exp(-height[kk + 1] / hh) * (theta[i, j, kk + 1] - t0[kk + 1]) / stat[kk + 1]
                altm = np.exp(-height[kk - 1] / hh) * (theta[i, j, kk - 1] - t0[kk - 1]) / stat[kk - 1]
                
                # Vertical derivative of the stretching term multiplied by zonal mean vorticity
                strc = (altp - altm) * zmav[j, kk] / (height[kk + 1] - height[kk - 1])
                
                # PV = absolute vorticity + stretching term
                pv[i, j, kk] = avort[i, j, kk] + np.exp(height[kk] / hh) * strc
    
    return pv


@njit(cache=True)
def _compute_qgpv_core(
    ut: np.ndarray,
    vt: np.ndarray,
    theta: np.ndarray,
    height: np.ndarray,
    t0: np.ndarray,
    stat: np.ndarray,
    aa: float,
    omega: float,
    dz: float,
    hh: float,
    rr: float,
    cp: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core implementation of compute_qgpv.
    
    This is the main computational kernel that computes quasi-geostrophic
    potential vorticity and absolute vorticity from input fields.
    
    Parameters
    ----------
    ut : np.ndarray
        Zonal wind field, shape (nlon, nlat, kmax)
    vt : np.ndarray
        Meridional wind field, shape (nlon, nlat, kmax)
    theta : np.ndarray
        Potential temperature, shape (nlon, nlat, kmax)
    height : np.ndarray
        Height levels, shape (kmax,)
    t0 : np.ndarray
        Reference temperature profile, shape (kmax,)
    stat : np.ndarray
        Static stability profile, shape (kmax,)
    aa : float
        Earth radius in meters
    omega : float
        Earth's rotation rate in rad/s
    dz : float
        Vertical grid spacing (not used in current implementation)
    hh : float
        Scale height in meters
    rr : float
        Dry gas constant (not used in current implementation)
    cp : float
        Specific heat at constant pressure (not used in current implementation)
        
    Returns
    -------
    pv : np.ndarray
        Quasi-geostrophic potential vorticity, shape (nlon, nlat, kmax)
    avort : np.ndarray
        Absolute vorticity, shape (nlon, nlat, kmax)
    """
    nlon, nlat, kmax = ut.shape
    
    # Constants
    pi = np.arccos(-1.0)
    dphi = pi / float(nlat - 1)
    # rkappa = rr / cp  # Not used
    
    # Step 1: Compute absolute vorticity
    avort = _compute_absolute_vorticity(ut, vt, nlon, nlat, kmax, aa, omega, dphi, pi)
    
    # Step 2: Compute zonal mean absolute vorticity
    zmav = _compute_zonal_mean_vorticity(avort, nlon, nlat, kmax)
    
    # Step 3: Compute interior PV with stretching term
    pv = _compute_interior_pv(avort, theta, zmav, height, t0, stat, nlon, nlat, kmax, omega, hh, pi)
    
    return pv, avort


def compute_qgpv(
    ut: np.ndarray,
    vt: np.ndarray,
    theta: np.ndarray,
    height: np.ndarray,
    t0: np.ndarray,
    stat: np.ndarray,
    aa: float,
    omega: float,
    dz: float,
    hh: float,
    rr: float,
    cp: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute quasi-geostrophic potential vorticity and absolute vorticity.
    
    This is a Numba-accelerated Python implementation of the Fortran subroutine
    `compute_qgpv`. It computes the quasi-geostrophic potential vorticity (QGPV)
    and absolute vorticity from the input wind and temperature fields.
    
    Parameters
    ----------
    ut : np.ndarray
        Zonal wind field with shape (nlon, nlat, kmax), in m/s.
        Must be a contiguous array.
    vt : np.ndarray
        Meridional wind field with shape (nlon, nlat, kmax), in m/s.
        Must be a contiguous array.
    theta : np.ndarray
        Potential temperature field with shape (nlon, nlat, kmax), in K.
        Must be a contiguous array.
    height : np.ndarray
        Height levels with shape (kmax,), in meters.
    t0 : np.ndarray
        Reference temperature profile with shape (kmax,), in K.
    stat : np.ndarray
        Static stability profile with shape (kmax,), in K/m.
    aa : float
        Earth radius in meters (typically 6.378e6).
    omega : float
        Earth's rotation rate in rad/s (typically 7.29e-5).
    dz : float
        Vertical grid spacing in meters. Note: This parameter is present
        for API compatibility but is not used in the calculation.
    hh : float
        Scale height in meters (typically 7000).
    rr : float
        Dry gas constant in J/(kg·K) (typically 287). Note: This parameter
        is present for API compatibility but is not used in the calculation.
    cp : float
        Specific heat at constant pressure in J/(kg·K) (typically 1004).
        Note: This parameter is present for API compatibility but is not
        used in the calculation.
        
    Returns
    -------
    pv : np.ndarray
        Quasi-geostrophic potential vorticity with shape (nlon, nlat, kmax),
        in 1/s. Values are only computed for interior vertical levels
        (indices 1 to kmax-2); boundary levels (0 and kmax-1) are set to zero.
    avort : np.ndarray
        Absolute vorticity with shape (nlon, nlat, kmax), in 1/s.
        
    Notes
    -----
    The QGPV is computed using the quasi-geostrophic approximation:
    
    .. math::
        q = \\zeta_a + \\frac{\\partial}{\\partial z}\\left(\\frac{f}{N^2}\\theta'\\right)
    
    where :math:`\\zeta_a` is the absolute vorticity, :math:`f` is the Coriolis
    parameter (approximated by zonal mean absolute vorticity), :math:`N^2` is
    the static stability, and :math:`\\theta'` is the potential temperature
    anomaly from the reference state.
    
    The absolute vorticity is computed as:
    
    .. math::
        \\zeta_a = 2\\Omega\\sin\\phi + \\frac{1}{a\\cos\\phi}\\frac{\\partial v}{\\partial \\lambda}
                 - \\frac{1}{a\\cos\\phi}\\frac{\\partial (u\\cos\\phi)}{\\partial \\phi}
    
    where :math:`\\Omega` is Earth's rotation rate, :math:`a` is Earth's radius,
    :math:`\\phi` is latitude, and :math:`\\lambda` is longitude.
    
    Periodic boundary conditions are applied in the longitudinal direction.
    Pole values are computed as the zonal average of the adjacent latitude.
    
    Examples
    --------
    >>> import numpy as np
    >>> from falwa.numba_modules import compute_qgpv
    >>> 
    >>> # Create sample input data
    >>> nlon, nlat, kmax = 144, 73, 17
    >>> ut = np.random.randn(nlon, nlat, kmax)
    >>> vt = np.random.randn(nlon, nlat, kmax)
    >>> theta = 300 + np.random.randn(nlon, nlat, kmax)
    >>> height = np.linspace(0, 16000, kmax)
    >>> t0 = np.linspace(300, 220, kmax)
    >>> stat = np.ones(kmax) * 0.01
    >>> 
    >>> # Compute QGPV
    >>> pv, avort = compute_qgpv(ut, vt, theta, height, t0, stat,
    ...                          aa=6.378e6, omega=7.29e-5, dz=1000,
    ...                          hh=7000, rr=287, cp=1004)
    
    See Also
    --------
    compute_qgpv_direct_inv : Alternative QGPV computation with direct inversion
    """
    # Ensure arrays are contiguous and float64
    ut = np.ascontiguousarray(ut, dtype=np.float64)
    vt = np.ascontiguousarray(vt, dtype=np.float64)
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    height = np.ascontiguousarray(height, dtype=np.float64)
    t0 = np.ascontiguousarray(t0, dtype=np.float64)
    stat = np.ascontiguousarray(stat, dtype=np.float64)
    
    return _compute_qgpv_core(ut, vt, theta, height, t0, stat, 
                              float(aa), float(omega), float(dz), 
                              float(hh), float(rr), float(cp))

