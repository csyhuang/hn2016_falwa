"""
Numba implementation of compute_qref_and_fawa_first.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `compute_qref_and_fawa_first` for computing reference QGPV (qref),
zonal-mean fields, and FAWA (Finite-Amplitude Wave Activity).

.. versionadded:: 2.4.0

Notes
-----
This implementation is equivalent to the Fortran subroutine in
`f90_modules/compute_qref_and_fawa_first.f90` and produces numerically identical results.
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True)
def _compute_zonal_means(
    pv: np.ndarray,
    uu: np.ndarray,
    pt: np.ndarray,
    imax: int,
    jmax: int,
    kmax: int,
    nd: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute zonal-mean fields for the hemisphere.
    
    Parameters
    ----------
    pv : np.ndarray
        Potential vorticity, shape (imax, jmax, kmax)
    uu : np.ndarray
        Zonal wind, shape (imax, jmax, kmax)
    pt : np.ndarray
        Potential temperature, shape (imax, jmax, kmax)
    imax : int
        Number of longitude points
    jmax : int
        Number of latitude points
    kmax : int
        Number of vertical levels
    nd : int
        Number of latitude points in the hemisphere
        
    Returns
    -------
    qbar : np.ndarray
        Zonal-mean PV, shape (nd, kmax)
    tbar : np.ndarray
        Zonal-mean potential temperature, shape (nd, kmax)
    ubar : np.ndarray
        Zonal-mean zonal wind, shape (nd, kmax)
    """
    qbar = np.zeros((nd, kmax), dtype=np.float64)
    tbar = np.zeros((nd, kmax), dtype=np.float64)
    ubar = np.zeros((nd, kmax), dtype=np.float64)
    
    # Fortran: do j = nd, jmax
    for j in range(nd, jmax + 1):  # 1-indexed in Fortran
        j_out = j - (nd - 1) - 1  # Convert to 0-indexed output (j-(nd-1) in Fortran, then -1 for 0-indexed)
        for k in range(kmax):
            for i in range(imax):
                qbar[j_out, k] += pv[i, j - 1, k] / float(imax)
                tbar[j_out, k] += pt[i, j - 1, k] / float(imax)
                ubar[j_out, k] += uu[i, j - 1, k] / float(imax)
    
    return qbar, tbar, ubar


@njit(cache=True)
def _area_analysis_qref(
    pv2: np.ndarray,
    imax: int,
    jmax: int,
    nd: int,
    nnd: int,
    a: float,
    dphi: float,
    dlambda: float,
    alat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute qref and cref via area analysis for one level.
    
    Parameters
    ----------
    pv2 : np.ndarray
        2D PV field for this level, shape (imax, jmax)
    imax : int
        Number of longitude points
    jmax : int
        Number of latitude points
    nd : int
        Number of latitude points in hemisphere
    nnd : int
        Number of partitions for area analysis
    a : float
        Earth radius
    dphi : float
        Latitude grid spacing in radians
    dlambda : float
        Longitude grid spacing in radians
    alat : np.ndarray
        Latitude area thresholds, shape (nd,)
        
    Returns
    -------
    qref_k : np.ndarray
        Reference PV for this level, shape (nd,)
    cref_k : np.ndarray
        Cumulative PV for this level, shape (nd,)
    """
    pi = np.arccos(-1.0)
    
    qref_k = np.zeros(nd, dtype=np.float64)
    cref_k = np.zeros(nd, dtype=np.float64)
    
    # Find min/max
    qmax = pv2[0, 0]
    qmin = pv2[0, 0]
    for j in range(jmax):
        for i in range(imax):
            if pv2[i, j] > qmax:
                qmax = pv2[i, j]
            if pv2[i, j] < qmin:
                qmin = pv2[i, j]
    
    dq = (qmax - qmin) / float(nnd - 1)
    
    # Initialize arrays
    qn = np.zeros(nnd, dtype=np.float64)
    an = np.zeros(nnd, dtype=np.float64)
    cn = np.zeros(nnd, dtype=np.float64)
    
    for nn in range(nnd):
        qn[nn] = qmax - dq * float(nn)
    
    # Area accumulation
    for j in range(jmax):
        phi0 = -0.5 * pi + dphi * float(j)
        for i in range(imax):
            if dq > 0:
                ind = int((qmax - pv2[i, j]) / dq)
            else:
                ind = 0
            if ind < 0:
                ind = 0
            if ind >= nnd:
                ind = nnd - 1
            da = a * a * dphi * dlambda * np.cos(phi0)
            an[ind] += da
            cn[ind] += da * pv2[i, j]
    
    # Cumulative sums
    aan = np.zeros(nnd, dtype=np.float64)
    ccn = np.zeros(nnd, dtype=np.float64)
    aan[0] = 0.0
    ccn[0] = 0.0
    for nn in range(1, nnd):
        aan[nn] = aan[nn - 1] + an[nn]
        ccn[nn] = ccn[nn - 1] + cn[nn]
    
    # Interpolate to get qref
    for j in range(nd - 1):  # Fortran: j = 1, nd-1
        for nn in range(nnd - 1):
            if aan[nn] <= alat[j] and aan[nn + 1] > alat[j]:
                dd = (alat[j] - aan[nn]) / (aan[nn + 1] - aan[nn])
                qref_k[j] = qn[nn] * (1.0 - dd) + qn[nn + 1] * dd
                cref_k[j] = ccn[nn] * (1.0 - dd) + ccn[nn + 1] * dd
    
    qref_k[nd - 1] = qmax  # Fortran: qref(nd, k) = qmax
    
    return qref_k, cref_k


@njit(cache=True)
def _area_analysis_ckref(
    vort2: np.ndarray,
    imax: int,
    jmax: int,
    nd: int,
    nnd: int,
    a: float,
    dphi: float,
    dlambda: float,
    alat: np.ndarray
) -> np.ndarray:
    """
    Compute Kelvin's circulation reference (ckref) via area analysis for one level.
    
    Parameters
    ----------
    vort2 : np.ndarray
        2D absolute vorticity field for this level, shape (imax, jmax)
    imax : int
        Number of longitude points
    jmax : int
        Number of latitude points
    nd : int
        Number of latitude points in hemisphere
    nnd : int
        Number of partitions for area analysis
    a : float
        Earth radius
    dphi : float
        Latitude grid spacing in radians
    dlambda : float
        Longitude grid spacing in radians
    alat : np.ndarray
        Latitude area thresholds, shape (nd,)
        
    Returns
    -------
    ckref_k : np.ndarray
        Reference Kelvin circulation for this level, shape (nd,)
    """
    pi = np.arccos(-1.0)
    
    ckref_k = np.zeros(nd, dtype=np.float64)
    
    # Find min/max
    qmax = vort2[0, 0]
    qmin = vort2[0, 0]
    for j in range(jmax):
        for i in range(imax):
            if vort2[i, j] > qmax:
                qmax = vort2[i, j]
            if vort2[i, j] < qmin:
                qmin = vort2[i, j]
    
    dq = (qmax - qmin) / float(nnd - 1)
    
    # Initialize arrays
    qn = np.zeros(nnd, dtype=np.float64)
    an = np.zeros(nnd, dtype=np.float64)
    cn = np.zeros(nnd, dtype=np.float64)
    
    for nn in range(nnd):
        qn[nn] = qmax - dq * float(nn)
    
    # Area accumulation
    for j in range(jmax):
        phi0 = -0.5 * pi + dphi * float(j)
        for i in range(imax):
            if dq > 0:
                ind = int((qmax - vort2[i, j]) / dq)
            else:
                ind = 0
            if ind < 0:
                ind = 0
            if ind >= nnd:
                ind = nnd - 1
            da = a * a * dphi * dlambda * np.cos(phi0)
            an[ind] += da
            cn[ind] += da * vort2[i, j]
    
    # Cumulative sums
    aan = np.zeros(nnd, dtype=np.float64)
    ccn = np.zeros(nnd, dtype=np.float64)
    aan[0] = 0.0
    ccn[0] = 0.0
    for nn in range(1, nnd):
        aan[nn] = aan[nn - 1] + an[nn]
        ccn[nn] = ccn[nn - 1] + cn[nn]
    
    # Interpolate to get ckref
    for j in range(nd - 1):  # Fortran: j = 1, nd-1
        for nn in range(nnd - 1):
            if aan[nn] <= alat[j] and aan[nn + 1] > alat[j]:
                dd = (alat[j] - aan[nn]) / (aan[nn + 1] - aan[nn])
                ckref_k[j] = ccn[nn] * (1.0 - dd) + ccn[nn + 1] * dd
    
    return ckref_k


@njit(cache=True)
def _compute_cbar(
    qbar: np.ndarray,
    nd: int,
    kmax: int,
    a: float,
    dphi: float
) -> np.ndarray:
    """
    Compute cbar for all levels.
    
    Parameters
    ----------
    qbar : np.ndarray
        Zonal-mean PV, shape (nd, kmax)
    nd : int
        Number of latitude points in hemisphere
    kmax : int
        Number of vertical levels
    a : float
        Earth radius
    dphi : float
        Latitude grid spacing in radians
        
    Returns
    -------
    cbar : np.ndarray
        shape (nd, kmax)
    """
    pi = np.arccos(-1.0)
    cbar = np.zeros((nd, kmax), dtype=np.float64)
    
    for k in range(1, kmax - 1):  # Fortran: k = 2, kmax-1
        cbar[nd - 1, k] = 0.0
        for j in range(nd - 2, -1, -1):  # Fortran: j = nd-1, 1, -1
            phi0 = dphi * (float(j + 1) - 0.5)  # j+1 because converting from Fortran 1-indexed
            cbar[j, k] = cbar[j + 1, k] + 0.5 * (qbar[j + 1, k] + qbar[j, k]) * \
                         a * dphi * 2.0 * pi * a * np.cos(phi0)
    
    return cbar


@njit(cache=True)
def _normalize_qref_by_sine(
    qref: np.ndarray,
    nd: int,
    kmax: int,
    dphi: float
) -> np.ndarray:
    """
    Normalize QGPV by sine of latitude.
    
    Parameters
    ----------
    qref : np.ndarray
        Reference PV, shape (nd, kmax)
    nd : int
        Number of latitude points in hemisphere
    kmax : int
        Number of vertical levels
    dphi : float
        Latitude grid spacing in radians
        
    Returns
    -------
    qref : np.ndarray
        Normalized reference PV, shape (nd, kmax)
    """
    # Normalize by sin(latitude)
    for j in range(1, nd):  # Fortran: j = 2, nd
        phi0 = dphi * float(j)
        cor = np.sin(phi0)
        for k in range(kmax):
            qref[j, k] = qref[j, k] / cor
    
    # Extrapolate to j=0
    for k in range(1, kmax - 1):  # Fortran: k = 2, kmax-1
        qref[0, k] = 2.0 * qref[1, k] - qref[2, k]
    
    return qref


@njit(cache=True)
def _compute_fawa(
    cref: np.ndarray,
    cbar: np.ndarray,
    nd: int,
    kmax: int,
    a: float
) -> np.ndarray:
    """
    Compute FAWA (Finite-Amplitude Wave Activity).
    
    Parameters
    ----------
    cref : np.ndarray
        Reference cumulative PV, shape (nd, kmax)
    cbar : np.ndarray
        Zonal-mean cumulative PV, shape (nd, kmax)
    nd : int
        Number of latitude points in hemisphere
    kmax : int
        Number of vertical levels
    a : float
        Earth radius
        
    Returns
    -------
    fawa : np.ndarray
        Finite-amplitude wave activity, shape (nd, kmax)
    """
    pi = np.arccos(-1.0)
    fawa = np.zeros((nd, kmax), dtype=np.float64)
    
    for k in range(kmax):
        for j in range(nd):
            fawa[j, k] = (cref[j, k] - cbar[j, k]) / (2.0 * pi * a)
    
    return fawa


@njit(cache=True)
def _compute_tjk_sjk(
    tbar: np.ndarray,
    jb: int,
    jd: int,
    nd: int,
    kmax: int,
    z: np.ndarray,
    omega: float,
    dz: float,
    h: float,
    dphi: float,
    a: float,
    rr: float,
    cp: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tjk and sjk arrays for the direct solver.
    
    Parameters
    ----------
    tbar : np.ndarray
        Zonal-mean temperature, shape (nd, kmax)
    jb : int
        Lower bounding latitude index (1-indexed)
    jd : int
        nd - lower bounding latitude
    nd : int
        Number of latitude points in hemisphere
    kmax : int
        Number of vertical levels
    z : np.ndarray
        Height levels, shape (kmax,)
    omega : float
        Earth rotation rate
    dz : float
        Vertical grid spacing
    h : float
        Scale height
    dphi : float
        Latitude grid spacing in radians
    a : float
        Earth radius
    rr : float
        Gas constant
    cp : float
        Specific heat
        
    Returns
    -------
    tjk : np.ndarray
        shape (jd-2, kmax-1)
    sjk : np.ndarray
        shape (jd-2, jd-2, kmax-1)
    """
    rkappa = rr / cp
    
    tjk = np.zeros((jd - 2, kmax - 1), dtype=np.float64)
    sjk = np.zeros((jd - 2, jd - 2, kmax - 1), dtype=np.float64)
    
    # Top boundary condition (Eqs. 24-25)
    # Fortran: do jj = jb+2, (nd-1)
    for jj in range(jb + 2, nd):  # jb+2 to nd-1 (inclusive) in Fortran 1-indexed
        j = jj - jb  # Fortran: j = jj - jb
        phi0 = float(jj - 1) * dphi
        cos0 = np.cos(phi0)
        sin0 = np.sin(phi0)
        
        # tjk(j-1, kmax-1) in Fortran -> tjk[j-2, kmax-2] in Python
        tjk_val = -dz * rr * cos0 * np.exp(-z[kmax - 2] * rkappa / h)
        tjk_val = tjk_val * (tbar[j, kmax - 1] - tbar[j - 2, kmax - 1])  # tbar(j+1,kmax) - tbar(j-1,kmax) in Fortran
        tjk_val = tjk_val / (4.0 * omega * sin0 * dphi * h * a)
        
        tjk[j - 2, kmax - 2] = tjk_val
        sjk[j - 2, j - 2, kmax - 2] = 1.0
    
    return tjk, sjk


@njit(cache=True)
def _compute_qref_and_fawa_first_core(
    pv: np.ndarray,
    uu: np.ndarray,
    vort: np.ndarray,
    pt: np.ndarray,
    tn0: np.ndarray,
    imax: int,
    jmax: int,
    kmax: int,
    nd: int,
    nnd: int,
    jb: int,
    jd: int,
    a: float,
    omega: float,
    dz: float,
    h: float,
    dphi: float,
    dlambda: float,
    rr: float,
    cp: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core implementation of compute_qref_and_fawa_first.
    """
    pi = np.arccos(-1.0)
    
    # Initialize output arrays
    qref = np.zeros((nd, kmax), dtype=np.float64)
    cref = np.zeros((nd, kmax), dtype=np.float64)
    ckref = np.zeros((nd, kmax), dtype=np.float64)
    
    # Setup latitude area thresholds
    phi = np.zeros(nd, dtype=np.float64)
    alat = np.zeros(nd, dtype=np.float64)
    for nn in range(nd):
        phi[nn] = dphi * float(nn)
        alat[nn] = 2.0 * pi * a * a * (1.0 - np.sin(phi[nn]))
    
    # Setup height array
    z = np.zeros(kmax, dtype=np.float64)
    for k in range(kmax):
        z[k] = dz * float(k)
    
    # Compute zonal means
    qbar, tbar, ubar = _compute_zonal_means(pv, uu, pt, imax, jmax, kmax, nd)
    
    # Use tn0 as hemispheric-mean potential temperature
    tb = tn0.copy()
    
    # Process each level
    for k in range(1, kmax - 1):  # Fortran: k = 2, kmax-1
        # Extract 2D fields for this level
        pv2 = np.zeros((imax, jmax), dtype=np.float64)
        vort2 = np.zeros((imax, jmax), dtype=np.float64)
        for j in range(jmax):
            for i in range(imax):
                pv2[i, j] = pv[i, j, k]
                vort2[i, j] = vort[i, j, k]
        
        # Area analysis for qref
        qref_k, cref_k = _area_analysis_qref(pv2, imax, jmax, nd, nnd, a, dphi, dlambda, alat)
        for j in range(nd):
            qref[j, k] = qref_k[j]
            cref[j, k] = cref_k[j]
        
        # Area analysis for ckref (Kelvin's circulation)
        ckref_k = _area_analysis_ckref(vort2, imax, jmax, nd, nnd, a, dphi, dlambda, alat)
        for j in range(nd):
            ckref[j, k] = ckref_k[j]
    
    # Compute cbar
    cbar = _compute_cbar(qbar, nd, kmax, a, dphi)
    
    # Normalize qref by sine
    qref = _normalize_qref_by_sine(qref, nd, kmax, dphi)
    
    # Compute FAWA
    fawa = _compute_fawa(cref, cbar, nd, kmax, a)
    
    # Compute tjk and sjk for direct solver
    tjk, sjk = _compute_tjk_sjk(tbar, jb, jd, nd, kmax, z, omega, dz, h, dphi, a, rr, cp)
    
    return qref, ubar, tbar, fawa, ckref, tjk, sjk


def compute_qref_and_fawa_first(
    pv: np.ndarray,
    uu: np.ndarray,
    vort: np.ndarray,
    pt: np.ndarray,
    tn0: np.ndarray,
    nd: int,
    nnd: int,
    jb: int,
    jd: int,
    a: float,
    omega: float,
    dz: float,
    h: float,
    dphi: float,
    dlambda: float,
    rr: float,
    cp: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reference QGPV, zonal-mean fields, FAWA, and direct solver arrays.
    
    This is a Numba-accelerated Python implementation of the Fortran subroutine
    `compute_qref_and_fawa_first`.
    
    Parameters
    ----------
    pv : np.ndarray
        Potential vorticity with shape (imax, jmax, kmax).
    uu : np.ndarray
        Zonal wind with shape (imax, jmax, kmax).
    vort : np.ndarray
        Absolute vorticity with shape (imax, jmax, kmax).
    pt : np.ndarray
        Potential temperature with shape (imax, jmax, kmax).
    tn0 : np.ndarray
        Reference temperature profile with shape (kmax,).
    nd : int
        Number of latitude points in the hemisphere.
    nnd : int
        Number of partitions for area analysis (typically = jmax).
    jb : int
        Lower bounding latitude index (1-indexed).
    jd : int
        nd - lower bounding latitude.
    a : float
        Earth radius in meters.
    omega : float
        Earth rotation rate in rad/s.
    dz : float
        Vertical grid spacing in meters.
    h : float
        Scale height in meters.
    dphi : float
        Latitude grid spacing in radians.
    dlambda : float
        Longitude grid spacing in radians.
    rr : float
        Gas constant in J/(kg·K).
    cp : float
        Specific heat at constant pressure in J/(kg·K).
        
    Returns
    -------
    qref : np.ndarray
        Reference QGPV normalized by sin(latitude), shape (nd, kmax).
    ubar : np.ndarray
        Zonal-mean zonal wind, shape (nd, kmax).
    tbar : np.ndarray
        Zonal-mean potential temperature, shape (nd, kmax).
    fawa : np.ndarray
        Finite-amplitude wave activity, shape (nd, kmax).
    ckref : np.ndarray
        Reference Kelvin circulation, shape (nd, kmax).
    tjk : np.ndarray
        Direct solver array, shape (jd-2, kmax-1).
    sjk : np.ndarray
        Direct solver array, shape (jd-2, jd-2, kmax-1).
        
    Notes
    -----
    This function computes:
    
    1. Zonal-mean fields (qbar, ubar, tbar)
    2. Reference QGPV (qref) via area analysis
    3. Reference Kelvin circulation (ckref) via area analysis on absolute vorticity
    4. FAWA (Finite-Amplitude Wave Activity)
    5. Arrays (tjk, sjk) for the direct inversion solver
    
    The qref is normalized by sin(latitude) before output.
    """
    # Ensure arrays are contiguous and float64
    pv = np.ascontiguousarray(pv, dtype=np.float64)
    uu = np.ascontiguousarray(uu, dtype=np.float64)
    vort = np.ascontiguousarray(vort, dtype=np.float64)
    pt = np.ascontiguousarray(pt, dtype=np.float64)
    tn0 = np.ascontiguousarray(tn0, dtype=np.float64)
    
    imax, jmax, kmax = pv.shape
    
    return _compute_qref_and_fawa_first_core(
        pv, uu, vort, pt, tn0,
        imax, jmax, kmax, nd, nnd, jb, jd,
        float(a), float(omega), float(dz), float(h),
        float(dphi), float(dlambda), float(rr), float(cp)
    )

