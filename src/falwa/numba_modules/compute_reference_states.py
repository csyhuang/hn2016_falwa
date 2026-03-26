"""
Numba implementation of compute_reference_states.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `compute_reference_states` for computing reference states
(qref, uref, tref) using an iterative SOR (Successive Over-Relaxation) solver.

.. versionadded:: 2.4.0

Notes
-----
This implementation is equivalent to the Fortran subroutine in
`f90_modules/compute_reference_states.f90` and produces numerically identical results.

Arrays use C-order indexing:
- 3D arrays: [k, j, i] where k=height, j=latitude, i=longitude
- 2D lat-height arrays: [k, j]
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True)
def _compute_zonal_means(
    pv: np.ndarray,
    uu: np.ndarray,
    pt: np.ndarray,
    nlon: int,
    nlat: int,
    kmax: int,
    jd: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute zonal-mean fields for the northern hemisphere.
    
    Parameters
    ----------
    pv : np.ndarray
        Potential vorticity, shape (kmax, nlat, nlon)
    uu : np.ndarray
        Zonal wind, shape (kmax, nlat, nlon)
    pt : np.ndarray
        Potential temperature, shape (kmax, nlat, nlon)
    nlon : int
        Number of longitude points
    nlat : int
        Number of latitude points
    kmax : int
        Number of vertical levels
    jd : int
        Latitude index for equatorward boundary
        
    Returns
    -------
    qbar : np.ndarray
        Zonal-mean PV, shape (kmax, jd)
    tbar : np.ndarray
        Zonal-mean potential temperature, shape (kmax, jd)
    ubar : np.ndarray
        Zonal-mean zonal wind, shape (kmax, jd)
    """
    qbar = np.zeros((kmax, jd), dtype=np.float64)
    tbar = np.zeros((kmax, jd), dtype=np.float64)
    ubar = np.zeros((kmax, jd), dtype=np.float64)
    
    for j in range(jd, nlat + 1):  # Fortran: j = jd, nlat
        j_out = j - (jd - 1) - 1  # Convert to 0-indexed output
        for k in range(kmax):
            for i in range(nlon):
                qbar[k, j_out] += pv[k, j - 1, i] / float(nlon)
                tbar[k, j_out] += pt[k, j - 1, i] / float(nlon)
                ubar[k, j_out] += uu[k, j - 1, i] / float(nlon)
    
    return qbar, tbar, ubar


@njit(cache=True)
def _compute_hemispheric_mean_temperature(
    tbar: np.ndarray,
    jd: int,
    nlat: int,
    kmax: int,
    dphi: float
) -> np.ndarray:
    """
    Compute hemispheric-mean potential temperature.
    
    Parameters
    ----------
    tbar : np.ndarray
        Zonal-mean potential temperature, shape (kmax, jd)
    jd : int
        Latitude index for equatorward boundary
    nlat : int
        Number of latitude points
    kmax : int
        Number of vertical levels
    dphi : float
        Latitude grid spacing in radians
        
    Returns
    -------
    tb : np.ndarray
        Hemispheric-mean potential temperature, shape (kmax,)
    """
    pi = np.arccos(-1.0)
    tb = np.zeros(kmax, dtype=np.float64)
    wt = 0.0
    
    for j in range(jd, nlat + 1):  # Fortran: j = jd, nlat
        phi0 = dphi * float(j - 1) - 0.5 * pi
        j_idx = j - (jd - 1) - 1  # 0-indexed
        for k in range(kmax):
            tb[k] += np.cos(phi0) * tbar[k, j_idx]
        wt += np.cos(phi0)
    
    for k in range(kmax):
        tb[k] = tb[k] / wt
    
    return tb


@njit(cache=True)
def _area_analysis(
    pv2: np.ndarray,
    nlon: int,
    nlat: int,
    jd: int,
    npart: int,
    a: float,
    dphi: float,
    dlambda: float,
    alat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform area analysis to compute qref and cref for one level.
    
    Parameters
    ----------
    pv2 : np.ndarray
        2D PV field, shape (nlat, nlon)
    nlon : int
        Number of longitude points
    nlat : int
        Number of latitude points
    jd : int
        Latitude index for equatorward boundary
    npart : int
        Number of partitions for area analysis
    a : float
        Earth radius
    dphi : float
        Latitude grid spacing in radians
    dlambda : float
        Longitude grid spacing in radians
    alat : np.ndarray
        Latitude area thresholds, shape (jd,)
        
    Returns
    -------
    qref_k : np.ndarray
        Reference PV for this level, shape (jd,)
    cref_k : np.ndarray
        Cumulative PV for this level, shape (jd,)
    """
    pi = np.arccos(-1.0)
    
    qref_k = np.zeros(jd, dtype=np.float64)
    cref_k = np.zeros(jd, dtype=np.float64)
    
    # Find min/max
    qmax = pv2[0, 0]
    qmin = pv2[0, 0]
    for j in range(nlat):
        for i in range(nlon):
            if pv2[j, i] > qmax:
                qmax = pv2[j, i]
            if pv2[j, i] < qmin:
                qmin = pv2[j, i]
    
    dq = (qmax - qmin) / float(npart - 1)
    
    # Initialize arrays
    qn = np.zeros(npart, dtype=np.float64)
    an = np.zeros(npart, dtype=np.float64)
    cn = np.zeros(npart, dtype=np.float64)
    
    for nn in range(npart):
        qn[nn] = qmax - dq * float(nn)
    
    # Area accumulation
    for j in range(nlat):
        phi0 = -0.5 * pi + dphi * float(j)
        for i in range(nlon):
            ind = int((qmax - pv2[j, i]) / dq)
            if ind < 0:
                ind = 0
            if ind >= npart:
                ind = npart - 1
            da = a * a * dphi * dlambda * np.cos(phi0)
            an[ind] += da
            cn[ind] += da * pv2[j, i]
    
    # Cumulative sums
    aan = np.zeros(npart, dtype=np.float64)
    ccn = np.zeros(npart, dtype=np.float64)
    aan[0] = 0.0
    ccn[0] = 0.0
    for nn in range(1, npart):
        aan[nn] = aan[nn - 1] + an[nn]
        ccn[nn] = ccn[nn - 1] + cn[nn]
    
    # Interpolate to get qref
    for j in range(jd - 1):  # Fortran: j = 1, jd-1
        for nn in range(npart - 1):
            if aan[nn] <= alat[j] and aan[nn + 1] > alat[j]:
                dd = (alat[j] - aan[nn]) / (aan[nn + 1] - aan[nn])
                qref_k[j] = qn[nn] * (1.0 - dd) + qn[nn + 1] * dd
                cref_k[j] = ccn[nn] * (1.0 - dd) + ccn[nn + 1] * dd
    
    qref_k[jd - 1] = qmax  # Fortran: qref(jd, k) = qmax
    
    return qref_k, cref_k


@njit(cache=True)
def _compute_cbar(
    qbar: np.ndarray,
    jd: int,
    kmax: int,
    a: float,
    dphi: float
) -> np.ndarray:
    """
    Compute cbar for all levels.
    
    Parameters
    ----------
    qbar : np.ndarray
        Zonal-mean PV, shape (kmax, jd)
    jd : int
        Latitude index for equatorward boundary
    kmax : int
        Number of vertical levels
    a : float
        Earth radius
    dphi : float
        Latitude grid spacing in radians
        
    Returns
    -------
    cbar : np.ndarray
        shape (kmax, jd)
    """
    pi = np.arccos(-1.0)
    cbar = np.zeros((kmax, jd), dtype=np.float64)
    
    for k in range(1, kmax - 1):  # Fortran: k = 2, kmax-1
        cbar[k, jd - 1] = 0.0
        for j in range(jd - 2, -1, -1):  # Fortran: j = jd-1, 1, -1
            phi0 = dphi * (float(j + 1) - 0.5)  # j+1 because 0-indexed
            cbar[k, j] = cbar[k, j + 1] + 0.5 * (qbar[k, j + 1] + qbar[k, j]) * \
                         a * dphi * 2.0 * pi * a * np.cos(phi0)
    
    return cbar


@njit(cache=True)
def _normalize_qref_by_coriolis(
    qref: np.ndarray,
    jd: int,
    kmax: int,
    om: float,
    dphi: float
) -> np.ndarray:
    """
    Normalize QGPV by the Coriolis parameter.
    
    Parameters
    ----------
    qref : np.ndarray
        Reference PV, shape (kmax, jd)
    jd : int
        Latitude index for equatorward boundary
    kmax : int
        Number of vertical levels
    om : float
        Earth rotation rate
    dphi : float
        Latitude grid spacing in radians
        
    Returns
    -------
    qref : np.ndarray
        Normalized reference PV, shape (kmax, jd)
    """
    for j in range(1, jd):  # Fortran: j = 2, jd
        phi0 = dphi * float(j)
        cor = 2.0 * om * np.sin(phi0)
        for k in range(kmax):
            qref[k, j] = qref[k, j] / cor
    
    # Extrapolate to j=0
    for k in range(1, kmax - 1):  # Fortran: k = 2, kmax-1
        qref[k, 0] = 2.0 * qref[k, 1] - qref[k, 2]
    
    return qref


@njit(cache=True)
def _setup_sor_coefficients(
    stat: np.ndarray,
    z: np.ndarray,
    jd: int,
    kmax: int,
    a: float,
    om: float,
    dz: float,
    h: float,
    r: float,
    cp: float,
    dphi: float,
    qref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Setup coefficients for the SOR elliptic solver.
    
    Returns
    -------
    ajk, bjk, cjk, djk, ejk, fjk : np.ndarray
        Coefficient arrays, each shape (kmax, jd)
    """
    rkappa = r / cp
    
    ajk = np.zeros((kmax, jd), dtype=np.float64)
    bjk = np.zeros((kmax, jd), dtype=np.float64)
    cjk = np.zeros((kmax, jd), dtype=np.float64)
    djk = np.zeros((kmax, jd), dtype=np.float64)
    ejk = np.zeros((kmax, jd), dtype=np.float64)
    fjk = np.zeros((kmax, jd), dtype=np.float64)
    
    for j in range(1, jd - 1):  # Fortran: j = 2, jd-1
        phi0 = float(j) * dphi
        phip = (float(j) + 0.5) * dphi
        phim = (float(j) - 0.5) * dphi
        cos0 = np.cos(phi0)
        cosp = np.cos(phip)
        cosm = np.cos(phim)
        sin0 = np.sin(phi0)
        sinp = np.sin(phip)
        sinm = np.sin(phim)
        
        for k in range(1, kmax - 1):  # Fortran: k = 2, kmax-1
            zp = 0.5 * (z[k + 1] + z[k])
            zm = 0.5 * (z[k - 1] + z[k])
            statp = 0.5 * (stat[k + 1] + stat[k])
            statm = 0.5 * (stat[k - 1] + stat[k])
            
            fact = 4.0 * om * om * h * a * a * sin0 * dphi * dphi / (dz * dz * r * cos0)
            
            amp = np.exp(-zp / h) * np.exp(rkappa * zp / h) / statp
            amp = amp * fact * np.exp(z[k] / h)
            
            amm = np.exp(-zm / h) * np.exp(rkappa * zm / h) / statm
            amm = amm * fact * np.exp(z[k] / h)
            
            ajk[k, j] = 1.0 / (sinp * cosp)
            bjk[k, j] = 1.0 / (sinm * cosm)
            cjk[k, j] = amp
            djk[k, j] = amm
            ejk[k, j] = -ajk[k, j] - bjk[k, j] - cjk[k, j] - djk[k, j]
            fjk[k, j] = -om * a * dphi * (qref[k, j + 1] - qref[k, j - 1])
    
    return ajk, bjk, cjk, djk, ejk, fjk


@njit(cache=True)
def _sor_solver(
    u: np.ndarray,
    ajk: np.ndarray,
    bjk: np.ndarray,
    cjk: np.ndarray,
    djk: np.ndarray,
    ejk: np.ndarray,
    fjk: np.ndarray,
    ubar: np.ndarray,
    tbar: np.ndarray,
    cref: np.ndarray,
    cbar: np.ndarray,
    z: np.ndarray,
    jd: int,
    kmax: int,
    maxits: int,
    eps: float,
    rjac: float,
    a: float,
    om: float,
    dz: float,
    h: float,
    r: float,
    cp: float,
    dphi: float
) -> Tuple[np.ndarray, int]:
    """
    SOR (Successive Over-Relaxation) elliptic solver.
    
    Parameters
    ----------
    u : np.ndarray
        Initial guess for solution, shape (kmax, jd)
    ajk, bjk, cjk, djk, ejk, fjk : np.ndarray
        Coefficient arrays from _setup_sor_coefficients
    ubar : np.ndarray
        Zonal-mean zonal wind, shape (kmax, jd)
    tbar : np.ndarray
        Zonal-mean potential temperature, shape (kmax, jd)
    cref : np.ndarray
        Reference cumulative PV, shape (kmax, jd)
    cbar : np.ndarray
        Cumulative zonal-mean PV, shape (kmax, jd)
    z : np.ndarray
        Height levels, shape (kmax,)
    jd : int
        Latitude index for equatorward boundary
    kmax : int
        Number of vertical levels
    maxits : int
        Maximum number of iterations
    eps : float
        Convergence tolerance
    rjac : float
        Spectral radius of Jacobi iteration
    a : float
        Earth radius
    om : float
        Earth rotation rate
    dz : float
        Vertical grid spacing
    h : float
        Scale height
    r : float
        Gas constant
    cp : float
        Specific heat
    dphi : float
        Latitude grid spacing in radians
        
    Returns
    -------
    u : np.ndarray
        Solution, shape (kmax, jd)
    num_of_iter : int
        Number of iterations performed
    """
    pi = np.arccos(-1.0)
    rkappa = r / cp
    
    # Compute anormf
    anormf = 0.0
    for j in range(1, jd - 1):
        for k in range(1, kmax - 1):
            anormf += np.abs(fjk[k, j])
    
    omega = 1.0
    converged = False
    num_of_iter = maxits
    
    for nnn in range(1, maxits + 1):
        anorm = 0.0
        
        for j in range(1, jd - 1):  # Fortran: j = 2, jd-1
            for k in range(1, kmax - 1):  # Fortran: k = 2, kmax-1
                if (j + k) % 2 == nnn % 2:
                    resid = ajk[k, j] * u[k, j + 1] + bjk[k, j] * u[k, j - 1] + \
                            cjk[k, j] * u[k + 1, j] + djk[k, j] * u[k - 1, j] + \
                            ejk[k, j] * u[k, j] - fjk[k, j]
                    anorm += np.abs(resid)
                    if ejk[k, j] != 0.0:
                        u[k, j] = u[k, j] - omega * resid / ejk[k, j]
                    else:
                        u[k, j] = 0.0
            
            # Boundary conditions
            u[0, j] = 0.0
            phi0 = dphi * float(j)
            uz = dz * r * np.cos(phi0) * np.exp(-z[kmax - 2] * rkappa / h)
            uz = uz * (tbar[kmax - 2, j + 1] - tbar[kmax - 2, j - 1]) / \
                 (2.0 * om * np.sin(phi0) * dphi * h * a)
            u[kmax - 1, j] = u[kmax - 3, j] - uz
        
        # Boundary conditions at j boundaries
        for k in range(kmax):
            u[k, jd - 1] = 0.0
            u[k, 0] = ubar[k, 0] + (cref[k, 0] - cbar[k, 0]) / (2.0 * pi * a)
        
        # Update omega
        if nnn == 1:
            omega = 1.0 / (1.0 - 0.5 * rjac * rjac)
        else:
            omega = 1.0 / (1.0 - 0.25 * rjac * rjac * omega)
        
        # Check convergence
        if nnn > 1 and anorm < eps * anormf:
            converged = True
            num_of_iter = nnn
            break
    
    # Handle non-convergence
    if not converged:
        # Set u to zero for non-converged case
        for j in range(jd):
            for k in range(kmax):
                u[k, j] = 0.0
    
    return u, num_of_iter


@njit(cache=True)
def _finalize_u(
    u: np.ndarray,
    ubar: np.ndarray,
    jd: int,
    kmax: int,
    dphi: float
) -> np.ndarray:
    """
    Finalize u by dividing by cos(phi) and setting boundary values.
    
    Parameters
    ----------
    u : np.ndarray
        Solution from SOR, shape (kmax, jd)
    ubar : np.ndarray
        Zonal-mean zonal wind, shape (kmax, jd)
    jd : int
        Latitude index
    kmax : int
        Number of vertical levels
    dphi : float
        Latitude grid spacing in radians
        
    Returns
    -------
    u : np.ndarray
        Finalized solution, shape (kmax, jd)
    """
    for j in range(1, jd - 1):  # Fortran: j = 2, jd-1
        phi0 = dphi * float(j)
        for k in range(kmax):
            u[k, j] = u[k, j] / np.cos(phi0)
    
    # Boundary conditions
    for k in range(kmax):
        u[k, 0] = ubar[k, 0]
        u[k, jd - 1] = 2.0 * u[k, jd - 2] - u[k, jd - 3]
    
    return u


@njit(cache=True)
def _compute_tref(
    u: np.ndarray,
    tb: np.ndarray,
    jd: int,
    kmax: int,
    om: float,
    dz: float,
    h: float,
    r: float,
    cp: float,
    dphi: float,
    a: float
) -> np.ndarray:
    """
    Compute reference temperature tref from u.
    
    Parameters
    ----------
    u : np.ndarray
        Reference wind, shape (kmax, jd)
    tb : np.ndarray
        Hemispheric-mean temperature, shape (kmax,)
    jd : int
        Latitude index
    kmax : int
        Number of vertical levels
    om : float
        Earth rotation rate
    dz : float
        Vertical grid spacing
    h : float
        Scale height
    r : float
        Gas constant
    cp : float
        Specific heat
    dphi : float
        Latitude grid spacing in radians
    a : float
        Earth radius
        
    Returns
    -------
    tref : np.ndarray
        Reference temperature, shape (kmax, jd)
    """
    rkappa = r / cp
    tref = np.zeros((kmax, jd), dtype=np.float64)
    tg = np.zeros(kmax, dtype=np.float64)
    
    for k in range(1, kmax - 1):  # Fortran: k = 2, kmax-1
        t00 = 0.0
        zz = dz * float(k)
        tref[k, 0] = t00
        tref[k, 1] = t00
        
        for j in range(1, jd - 1):  # Fortran: j = 2, jd-1
            phi0 = dphi * float(j)
            cor = 2.0 * om * np.sin(phi0)
            uz = (u[k + 1, j] - u[k - 1, j]) / (2.0 * dz)
            ty = -cor * uz * a * h * np.exp(rkappa * zz / h)
            ty = ty / r
            tref[k, j + 1] = tref[k, j - 1] + 2.0 * ty * dphi
        
        # Compute tg for this level
        tg[k] = 0.0
        wt = 0.0
        for j in range(jd):
            phi0 = dphi * float(j)
            tg[k] += np.cos(phi0) * tref[k, j]
            wt += np.cos(phi0)
        tg[k] = tg[k] / wt
        
        # Adjust tref
        tres = tb[k] - tg[k]
        for j in range(jd):
            tref[k, j] = tref[k, j] + tres
    
    # Boundary levels
    for j in range(jd):
        tref[0, j] = tref[1, j] - tb[1] + tb[0]
        tref[kmax - 1, j] = tref[kmax - 2, j] - tb[kmax - 2] + tb[kmax - 1]
    
    return tref


@njit(cache=True)
def _compute_reference_states_core(
    pv: np.ndarray,
    uu: np.ndarray,
    pt: np.ndarray,
    stat: np.ndarray,
    nlon: int,
    nlat: int,
    kmax: int,
    jd: int,
    npart: int,
    maxits: int,
    a: float,
    om: float,
    dz: float,
    eps: float,
    h: float,
    dphi: float,
    dlambda: float,
    r: float,
    cp: float,
    rjac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Core implementation of compute_reference_states.
    """
    pi = np.arccos(-1.0)
    
    # Initialize arrays (C-order: [k, j])
    qref = np.zeros((kmax, jd), dtype=np.float64)
    u = np.zeros((kmax, jd), dtype=np.float64)
    tref = np.zeros((kmax, jd), dtype=np.float64)
    cref = np.zeros((kmax, jd), dtype=np.float64)
    
    # Setup latitude area thresholds
    phi = np.zeros(jd, dtype=np.float64)
    alat = np.zeros(jd, dtype=np.float64)
    for nn in range(jd):
        phi[nn] = dphi * float(nn)
        alat[nn] = 2.0 * pi * a * a * (1.0 - np.sin(phi[nn]))
    
    # Setup height array
    z = np.zeros(kmax, dtype=np.float64)
    for k in range(kmax):
        z[k] = dz * float(k)
    
    # Compute zonal means
    qbar, tbar, ubar = _compute_zonal_means(pv, uu, pt, nlon, nlat, kmax, jd)
    
    # Compute hemispheric-mean temperature
    tb = _compute_hemispheric_mean_temperature(tbar, jd, nlat, kmax, dphi)
    
    # Area analysis for each level
    for k in range(1, kmax - 1):
        pv2 = np.zeros((nlat, nlon), dtype=np.float64)
        for j in range(nlat):
            for i in range(nlon):
                pv2[j, i] = pv[k, j, i]
        
        qref_k, cref_k = _area_analysis(pv2, nlon, nlat, jd, npart, a, dphi, dlambda, alat)
        for j in range(jd):
            qref[k, j] = qref_k[j]
            cref[k, j] = cref_k[j]
    
    # Compute cbar
    cbar = _compute_cbar(qbar, jd, kmax, a, dphi)
    
    # Normalize qref by Coriolis parameter
    qref = _normalize_qref_by_coriolis(qref, jd, kmax, om, dphi)
    
    # Setup SOR coefficients
    ajk, bjk, cjk, djk, ejk, fjk = _setup_sor_coefficients(
        stat, z, jd, kmax, a, om, dz, h, r, cp, dphi, qref
    )
    
    # SOR solver
    u, num_of_iter = _sor_solver(
        u, ajk, bjk, cjk, djk, ejk, fjk, ubar, tbar, cref, cbar, z,
        jd, kmax, maxits, eps, rjac, a, om, dz, h, r, cp, dphi
    )
    
    # Finalize u
    u = _finalize_u(u, ubar, jd, kmax, dphi)
    
    # Compute tref
    tref = _compute_tref(u, tb, jd, kmax, om, dz, h, r, cp, dphi, a)
    
    return qref, u, tref, num_of_iter


def compute_reference_states(
    pv: np.ndarray,
    uu: np.ndarray,
    pt: np.ndarray,
    stat: np.ndarray,
    jd: int,
    npart: int,
    maxits: int,
    a: float,
    om: float,
    dz: float,
    eps: float,
    h: float,
    dphi: float,
    dlambda: float,
    r: float,
    cp: float,
    rjac: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute reference states (qref, uref, tref) using SOR elliptic solver.
    
    This is a Numba-accelerated Python implementation of the Fortran subroutine
    `compute_reference_states`.
    
    Parameters
    ----------
    pv : np.ndarray
        Potential vorticity with shape (kmax, nlat, nlon).
    uu : np.ndarray
        Zonal wind with shape (kmax, nlat, nlon).
    pt : np.ndarray
        Potential temperature with shape (kmax, nlat, nlon).
    stat : np.ndarray
        Static stability with shape (kmax,).
    jd : int
        Latitude index for equatorward boundary.
    npart : int
        Number of partitions for area analysis.
    maxits : int
        Maximum number of SOR iterations.
    a : float
        Earth radius in meters.
    om : float
        Earth rotation rate in rad/s.
    dz : float
        Vertical grid spacing in meters.
    eps : float
        Convergence tolerance.
    h : float
        Scale height in meters.
    dphi : float
        Latitude grid spacing in radians.
    dlambda : float
        Longitude grid spacing in radians.
    r : float
        Gas constant in J/(kg·K).
    cp : float
        Specific heat at constant pressure in J/(kg·K).
    rjac : float
        Spectral radius of Jacobi iteration (typically 0.95).
        
    Returns
    -------
    qref : np.ndarray
        Reference potential vorticity with shape (kmax, jd).
    u : np.ndarray
        Reference zonal wind with shape (kmax, jd).
    tref : np.ndarray
        Reference temperature with shape (kmax, jd).
    num_of_iter : int
        Number of iterations performed by the SOR solver.
        
    Notes
    -----
    This function computes the reference states for one hemisphere using
    an iterative SOR (Successive Over-Relaxation) elliptic solver.
    The algorithm consists of:
    
    1. Compute zonal-mean fields
    2. Compute hemispheric-mean potential temperature
    3. Area analysis to determine qref
    4. Normalize qref by Coriolis parameter
    5. SOR solver to compute reference wind field
    6. Compute reference temperature from thermal wind balance
    
    If the SOR solver does not converge within maxits iterations,
    the reference wind is set to zero.
    """
    kmax, nlat, nlon = pv.shape
    
    return _compute_reference_states_core(
        pv, uu, pt, stat, nlon, nlat, kmax, jd, npart, maxits,
        float(a), float(om), float(dz), float(eps), float(h),
        float(dphi), float(dlambda), float(r), float(cp), float(rjac)
    )
