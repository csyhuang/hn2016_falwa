"""
Numba implementation of compute_qgpv_direct_inv.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `compute_qgpv_direct_inv` for computing quasi-geostrophic potential
vorticity and absolute vorticity with hemisphere-dependent reference states.

.. versionadded:: 2.4.0

.. versionchanged:: 2.4.0
   Added explicit type signatures for eager compilation at import time.

Notes
-----
This implementation is equivalent to the Fortran subroutine in
`f90_modules/compute_qgpv_direct_inv.f90` and produces numerically identical results.
The function is JIT-compiled using Numba for performance comparable to
the original Fortran implementation.

The key difference from `compute_qgpv` is that this version uses separate
reference temperature and static stability profiles for the southern and
northern hemispheres, with the equator boundary specified by `jd`.

Arrays use C-order indexing:
- 3D arrays: [k, j, i] where k=height, j=latitude, i=longitude
- 2D lat-height arrays: [k, j]
"""

import numpy as np
from numba import njit, float64, int64
from numba.core.types import Tuple as NbTuple
from typing import Tuple

# Type aliases for readability
f8 = float64
i8 = int64
f8_1d = float64[:]
f8_2d = float64[:, :]
f8_3d = float64[:, :, :]


@njit(f8_3d(f8_3d, f8_3d, i8, i8, i8, f8, f8, f8, f8), cache=True)
def _compute_absolute_vorticity_direct_inv(
    uq: np.ndarray,
    vq: np.ndarray,
    nlon: int,
    nlat: int,
    kmax: int,
    aa: float,
    omega: float,
    dphi: float,
    pi: float
) -> np.ndarray:
    """
    Compute absolute vorticity from wind fields (direct inversion version).
    
    Parameters
    ----------
    uq : np.ndarray
        Zonal wind field, shape (kmax, nlat, nlon)
    vq : np.ndarray
        Meridional wind field, shape (kmax, nlat, nlon)
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
        Absolute vorticity, shape (kmax, nlat, nlon)
    """
    avort = np.zeros((kmax, nlat, nlon), dtype=np.float64)
    
    for kk in range(kmax):
        # Interior latitudes (j=1 to nlat-2 in 0-indexed)
        for j in range(1, nlat - 1):
            # Fortran: phi0 = -0.5*pi + pi*float(j-1)/float(nlat-1)
            # In Python with 0-indexed j, this becomes:
            phi0 = -0.5 * pi + pi * float(j) / float(nlat - 1)
            phim = -0.5 * pi + pi * float(j - 1) / float(nlat - 1)
            phip = -0.5 * pi + pi * float(j + 1) / float(nlat - 1)
            
            cos_phi0 = np.cos(phi0)
            cos_phim = np.cos(phim)
            cos_phip = np.cos(phip)
            
            # Planetary vorticity
            av1 = 2.0 * omega * np.sin(phi0)
            
            # Interior longitudes (i=1 to nlon-2 in 0-indexed)
            for i in range(1, nlon - 1):
                av2 = (vq[kk, j, i + 1] - vq[kk, j, i - 1]) / (2.0 * aa * cos_phi0 * dphi)
                av3 = -(uq[kk, j + 1, i] * cos_phip - uq[kk, j - 1, i] * cos_phim) / (2.0 * aa * cos_phi0 * dphi)
                avort[kk, j, i] = av1 + av2 + av3
            
            # Periodic boundary conditions for i=0 (Fortran i=1)
            av2 = (vq[kk, j, 1] - vq[kk, j, nlon - 1]) / (2.0 * aa * cos_phi0 * dphi)
            av3 = -(uq[kk, j + 1, 0] * cos_phip - uq[kk, j - 1, 0] * cos_phim) / (2.0 * aa * cos_phi0 * dphi)
            avort[kk, j, 0] = av1 + av2 + av3
            
            # Periodic boundary conditions for i=nlon-1 (Fortran i=nlon)
            av5 = (vq[kk, j, 0] - vq[kk, j, nlon - 2]) / (2.0 * aa * cos_phi0 * dphi)
            av6 = -(uq[kk, j + 1, nlon - 1] * cos_phip - uq[kk, j - 1, nlon - 1] * cos_phim) / (2.0 * aa * cos_phi0 * dphi)
            avort[kk, j, nlon - 1] = av1 + av5 + av6
        
        # Pole values: average from neighboring latitude
        avs = 0.0
        avn = 0.0
        for i in range(nlon):
            avs += avort[kk, 1, i] / float(nlon)
            avn += avort[kk, nlat - 2, i] / float(nlon)
        
        # Set pole values (all longitudes get the same value)
        for i in range(nlon):
            avort[kk, 0, i] = avs       # South pole
            avort[kk, nlat - 1, i] = avn  # North pole
    
    return avort


@njit(f8_2d(f8_3d, i8, i8, i8), cache=True)
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
        Absolute vorticity, shape (kmax, nlat, nlon)
    nlon : int
        Number of longitude points
    nlat : int
        Number of latitude points
    kmax : int
        Number of vertical levels
        
    Returns
    -------
    zmav : np.ndarray
        Zonal mean absolute vorticity, shape (kmax, nlat)
    """
    zmav = np.zeros((kmax, nlat), dtype=np.float64)
    
    for kk in range(kmax):
        for j in range(nlat):
            for i in range(nlon):
                zmav[kk, j] += avort[kk, j, i] / float(nlon)
    
    return zmav


@njit(f8_3d(f8_3d, f8_3d, f8_1d, f8_1d, f8_1d, f8_1d, f8_1d, i8, i8, i8, i8, f8, f8, f8), cache=True)
def _compute_interior_pv_direct_inv(
    avort: np.ndarray,
    tq: np.ndarray,
    height: np.ndarray,
    ts0: np.ndarray,
    tn0: np.ndarray,
    stats: np.ndarray,
    statn: np.ndarray,
    nlon: int,
    nlat: int,
    kmax: int,
    jd: int,
    omega: float,
    hh: float,
    pi: float
) -> np.ndarray:
    """
    Compute interior potential vorticity with hemisphere-dependent stretching term.
    
    Parameters
    ----------
    avort : np.ndarray
        Absolute vorticity, shape (kmax, nlat, nlon)
    tq : np.ndarray
        Potential temperature, shape (kmax, nlat, nlon)
    height : np.ndarray
        Height levels, shape (kmax,)
    ts0 : np.ndarray
        Southern hemisphere reference temperature, shape (kmax,)
    tn0 : np.ndarray
        Northern hemisphere reference temperature, shape (kmax,)
    stats : np.ndarray
        Southern hemisphere static stability, shape (kmax,)
    statn : np.ndarray
        Northern hemisphere static stability, shape (kmax,)
    nlon : int
        Number of longitude points
    nlat : int
        Number of latitude points
    kmax : int
        Number of vertical levels
    jd : int
        Equatorial boundary index (1-indexed as in Fortran)
    omega : float
        Earth's rotation rate
    hh : float
        Scale height
    pi : float
        Pi constant
        
    Returns
    -------
    pv : np.ndarray
        Potential vorticity, shape (kmax, nlat, nlon)
    """
    pv = np.zeros((kmax, nlat, nlon), dtype=np.float64)
    
    # Interior levels only (kk=1 to kmax-2 in 0-indexed)
    for kk in range(1, kmax - 1):
        for j in range(nlat):
            phi0 = -0.5 * pi + pi * float(j) / float(nlat - 1)
            f = 2.0 * omega * np.sin(phi0)
            
            # Select hemisphere-dependent parameters
            # Fortran uses 1-indexed j, so j <= jd means southern hemisphere
            # In Python 0-indexed, j+1 <= jd means j < jd (since j+1 is the Fortran index)
            if (j + 1) <= jd:  # Southern hemisphere
                statp = stats[kk + 1]
                statm = stats[kk - 1]
                t00p = ts0[kk + 1]
                t00m = ts0[kk - 1]
            else:  # Northern hemisphere
                statp = statn[kk + 1]
                statm = statn[kk - 1]
                t00p = tn0[kk + 1]
                t00m = tn0[kk - 1]
            
            for i in range(nlon):
                thetap = tq[kk + 1, j, i]
                thetam = tq[kk - 1, j, i]
                altp = np.exp(-height[kk + 1] / hh) * (thetap - t00p) / statp
                altm = np.exp(-height[kk - 1] / hh) * (thetam - t00m) / statm
                strc = (altp - altm) * f / (height[kk + 1] - height[kk - 1])
                pv[kk, j, i] = avort[kk, j, i] + np.exp(height[kk] / hh) * strc
    
    return pv


@njit(NbTuple((f8_3d, f8_3d))(f8_3d, f8_3d, f8_3d, f8_1d, f8_1d, f8_1d, f8_1d, f8_1d, i8, f8, f8, f8, f8, f8, f8), cache=True)
def _compute_qgpv_direct_inv_core(
    uq: np.ndarray,
    vq: np.ndarray,
    tq: np.ndarray,
    height: np.ndarray,
    ts0: np.ndarray,
    tn0: np.ndarray,
    stats: np.ndarray,
    statn: np.ndarray,
    jd: int,
    aa: float,
    omega: float,
    dz: float,
    hh: float,
    rr: float,
    cp: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core implementation of compute_qgpv_direct_inv.
    
    This is the main computational kernel that computes quasi-geostrophic
    potential vorticity and absolute vorticity from input fields with
    hemisphere-dependent reference states.
    
    Parameters
    ----------
    uq : np.ndarray
        Zonal wind field, shape (kmax, nlat, nlon)
    vq : np.ndarray
        Meridional wind field, shape (kmax, nlat, nlon)
    tq : np.ndarray
        Potential temperature field, shape (kmax, nlat, nlon)
    height : np.ndarray
        Height levels, shape (kmax,)
    ts0 : np.ndarray
        Southern hemisphere reference temperature profile, shape (kmax,)
    tn0 : np.ndarray
        Northern hemisphere reference temperature profile, shape (kmax,)
    stats : np.ndarray
        Southern hemisphere static stability profile, shape (kmax,)
    statn : np.ndarray
        Northern hemisphere static stability profile, shape (kmax,)
    jd : int
        Equatorial boundary index (1-indexed as in Fortran)
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
        Quasi-geostrophic potential vorticity, shape (kmax, nlat, nlon)
    avort : np.ndarray
        Absolute vorticity, shape (kmax, nlat, nlon)
    """
    kmax, nlat, nlon = uq.shape
    
    # Constants
    pi = np.arccos(-1.0)
    dphi = pi / float(nlat - 1)
    
    # Step 1: Compute absolute vorticity
    avort = _compute_absolute_vorticity_direct_inv(uq, vq, nlon, nlat, kmax, aa, omega, dphi, pi)
    
    # Step 2: Compute zonal mean absolute vorticity (computed but not used in PV calculation here)
    # zmav = _compute_zonal_mean_vorticity(avort, nlon, nlat, kmax)
    
    # Step 3: Compute interior PV with hemisphere-dependent stretching term
    pv = _compute_interior_pv_direct_inv(avort, tq, height, ts0, tn0, stats, statn,
                                          nlon, nlat, kmax, jd, omega, hh, pi)
    
    return pv, avort


def compute_qgpv_direct_inv(
    jd: int,
    uq: np.ndarray,
    vq: np.ndarray,
    tq: np.ndarray,
    height: np.ndarray,
    ts0: np.ndarray,
    tn0: np.ndarray,
    stats: np.ndarray,
    statn: np.ndarray,
    aa: float,
    omega: float,
    dz: float,
    hh: float,
    rr: float,
    cp: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute quasi-geostrophic potential vorticity with hemisphere-dependent reference states.
    
    This is a Numba-accelerated Python implementation of the Fortran subroutine
    `compute_qgpv_direct_inv`. It computes the quasi-geostrophic potential vorticity
    (QGPV) and absolute vorticity using separate reference states for the southern
    and northern hemispheres.
    
    Parameters
    ----------
    jd : int
        Equatorial boundary index (1-indexed). Grid points with j <= jd use
        southern hemisphere reference states; j > jd use northern hemisphere.
    uq : np.ndarray
        Zonal wind field with shape (kmax, nlat, nlon), in m/s.
    vq : np.ndarray
        Meridional wind field with shape (kmax, nlat, nlon), in m/s.
    tq : np.ndarray
        Potential temperature field with shape (kmax, nlat, nlon), in K.
    height : np.ndarray
        Height levels with shape (kmax,), in meters.
    ts0 : np.ndarray
        Southern hemisphere reference temperature profile with shape (kmax,), in K.
    tn0 : np.ndarray
        Northern hemisphere reference temperature profile with shape (kmax,), in K.
    stats : np.ndarray
        Southern hemisphere static stability profile with shape (kmax,), in K/m.
    statn : np.ndarray
        Northern hemisphere static stability profile with shape (kmax,), in K/m.
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
        Specific heat in J/(kg·K) (typically 1004).
        Note: This parameter is present for API compatibility but is not
        used in the calculation.
        
    Returns
    -------
    pv : np.ndarray
        Quasi-geostrophic potential vorticity with shape (kmax, nlat, nlon),
        in 1/s. Values are only computed for interior vertical levels
        (indices 1 to kmax-2); boundary levels (0 and kmax-1) are set to zero.
    avort : np.ndarray
        Absolute vorticity with shape (kmax, nlat, nlon), in 1/s.
        
    Notes
    -----
    This version differs from `compute_qgpv` in that it uses hemisphere-dependent
    reference states. The stretching term in the QGPV calculation uses:
    
    - For j <= jd (southern hemisphere): ts0 and stats
    - For j > jd (northern hemisphere): tn0 and statn
    
    The QGPV is computed as:
    
    .. math::
        q = \\zeta_a + \\frac{\\partial}{\\partial z}\\left(\\frac{f}{N^2}\\theta'\\right)
    
    where the Coriolis parameter f is used directly (not the zonal mean vorticity
    as in `compute_qgpv`).
    
    Examples
    --------
    >>> import numpy as np
    >>> from falwa.numba_modules import compute_qgpv_direct_inv
    >>> 
    >>> # Create sample input data
    >>> kmax, nlat, nlon = 17, 73, 144
    >>> uq = np.random.randn(kmax, nlat, nlon)
    >>> vq = np.random.randn(kmax, nlat, nlon)
    >>> tq = 300 + np.random.randn(kmax, nlat, nlon)
    >>> height = np.linspace(0, 16000, kmax)
    >>> ts0 = np.linspace(300, 220, kmax)  # Southern hemisphere
    >>> tn0 = np.linspace(300, 220, kmax)  # Northern hemisphere
    >>> stats = np.ones(kmax) * 0.01
    >>> statn = np.ones(kmax) * 0.01
    >>> jd = nlat // 2  # Equator at midpoint
    >>> 
    >>> # Compute QGPV
    >>> pv, avort = compute_qgpv_direct_inv(jd, uq, vq, tq, height, ts0, tn0,
    ...                                      stats, statn,
    ...                                      aa=6.378e6, omega=7.29e-5, dz=1000,
    ...                                      hh=7000, rr=287, cp=1004)
    
    See Also
    --------
    compute_qgpv : QGPV computation with single reference state
    """
    return _compute_qgpv_direct_inv_core(
        uq, vq, tq, height, ts0, tn0, stats, statn,
        int(jd), float(aa), float(omega), float(dz),
        float(hh), float(rr), float(cp)
    )
