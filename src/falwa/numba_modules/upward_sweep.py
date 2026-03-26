"""
Numba implementation of upward_sweep.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `upward_sweep` for the upward sweep step in the direct solver
algorithm to recover u, tref, and qref.

.. versionadded:: 2.4.0

Notes
-----
Arrays use C-order indexing:
- 2D lat-height arrays: [k, j]
- 3D arrays: [k, j1, j2]
"""

import numpy as np
from numba import njit
from typing import Tuple


@njit(cache=True)
def _upward_sweep_core(
    jmax: int,
    kmax: int,
    nd: int,
    jb: int,
    jd: int,
    sjk: np.ndarray,
    tjk: np.ndarray,
    ckref: np.ndarray,
    tb: np.ndarray,
    qref_over_cor: np.ndarray,
    a: float,
    om: float,
    dz: float,
    h: float,
    rr: float,
    cp: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core implementation of upward_sweep.
    """
    rkappa = rr / cp
    pi = np.arccos(-1.0)
    dp = pi / float(jmax - 1)
    
    # Initialize arrays (C-order: [k, j])
    pjk = np.zeros((kmax, jd - 2), dtype=np.float64)
    u = np.zeros((kmax, jd), dtype=np.float64)
    tref = np.zeros((kmax, jd), dtype=np.float64)
    qref = np.zeros((kmax, nd), dtype=np.float64)
    tg = np.zeros(kmax, dtype=np.float64)
    
    # pjk[0, :] = 0. (already initialized)
    
    # Upward sweep: Fortran k = 1, kmax-1
    for k in range(kmax - 1):
        pj = np.zeros(jd - 2, dtype=np.float64)
        for i in range(jd - 2):
            pj[i] = pjk[k, i]
        
        sjj = np.zeros((jd - 2, jd - 2), dtype=np.float64)
        for i in range(jd - 2):
            for j in range(jd - 2):
                sjj[i, j] = sjk[k, i, j]
        
        tj = np.zeros(jd - 2, dtype=np.float64)
        for i in range(jd - 2):
            tj[i] = tjk[k, i]
        
        yj = np.zeros(jd - 2, dtype=np.float64)
        for i in range(jd - 2):
            yj[i] = 0.0
            for kk in range(jd - 2):
                yj[i] += sjj[i, kk] * pj[kk]
            pjk[k + 1, i] = yj[i] + tj[i]
    
    # Recover u: Fortran k = 1, kmax; j = 2, jd-1
    for k in range(kmax):
        for j in range(1, jd - 1):
            u[k, j] = pjk[k, j - 1]
    
    # Corner boundary conditions
    u[0, 0] = 0.0
    u[0, jd - 1] = 0.0
    u[kmax - 1, 0] = ckref[kmax - 1, jb] / (2.0 * pi * a) - om * a * np.cos(dp * float(jb))
    u[kmax - 1, jd - 1] = 0.0
    
    # Divide by cos phi to recover Uref
    for jj in range(jb + 1, nd):
        j = jj - jb
        phi0 = dp * float(jj - 1)
        for k in range(kmax):
            u[k, j - 1] = u[k, j - 1] / np.cos(phi0)
    
    # Extrapolate u at jd
    for k in range(kmax):
        u[k, jd - 1] = 2.0 * u[k, jd - 2] - u[k, jd - 3]
    
    # Copy qref_over_cor to qref
    for j in range(nd):
        for k in range(kmax):
            qref[k, j] = qref_over_cor[k, j]
    
    # Compute tref: Fortran k = 2, kmax-1
    for k in range(1, kmax - 1):
        t00 = 0.0
        zz = dz * float(k)
        tref[k, 0] = t00
        tref[k, 1] = t00
        
        for j in range(1, jd - 1):
            phi0 = dp * float(j + jb)
            cor = 2.0 * om * np.sin(phi0)
            uz = (u[k + 1, j] - u[k - 1, j]) / (2.0 * dz)
            ty = -cor * uz * a * h * np.exp(rkappa * zz / h)
            ty = ty / rr
            tref[k, j + 1] = tref[k, j - 1] + 2.0 * ty * dp
        
        # Compute qref from qref_over_cor
        for j in range(nd):
            phi0 = dp * float(j)
            qref[k, j] = qref_over_cor[k, j] * np.sin(phi0)
        
        # Compute tg (area-weighted mean)
        tg[k] = 0.0
        wt = 0.0
        for jj in range(jb + 1, nd + 1):
            j = jj - jb
            phi0 = dp * float(jj - 1)
            tg[k] += np.cos(phi0) * tref[k, j - 1]
            wt += np.cos(phi0)
        tg[k] = tg[k] / wt
        
        tres = tb[k] - tg[k]
        for j in range(jd):
            tref[k, j] = tref[k, j] + tres
    
    # Boundary levels
    for j in range(jd):
        tref[0, j] = tref[1, j] - tb[1] + tb[0]
        tref[kmax - 1, j] = tref[kmax - 2, j] - tb[kmax - 2] + tb[kmax - 1]
    
    return tref, qref, u


def upward_sweep(
    jmax: int,
    jb: int,
    sjk: np.ndarray,
    tjk: np.ndarray,
    ckref: np.ndarray,
    tb: np.ndarray,
    qref_over_cor: np.ndarray,
    a: float,
    om: float,
    dz: float,
    h: float,
    rr: float,
    cp: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform upward sweep to recover u, tref, and qref.
    
    Parameters
    ----------
    jmax : int
        Number of latitude points.
    jb : int
        Lower bounding latitude index.
    sjk : np.ndarray
        S matrix array, shape (kmax-1, jd-2, jd-2).
    tjk : np.ndarray
        T vector array, shape (kmax-1, jd-2).
    ckref : np.ndarray
        Reference Kelvin circulation, shape (kmax, nd).
    tb : np.ndarray
        Hemispheric-mean temperature, shape (kmax,).
    qref_over_cor : np.ndarray
        Reference QGPV normalized by sin(lat), shape (kmax, nd).
    a : float
        Earth radius.
    om : float
        Earth rotation rate.
    dz : float
        Vertical grid spacing.
    h : float
        Scale height.
    rr : float
        Gas constant.
    cp : float
        Specific heat.
        
    Returns
    -------
    tref : np.ndarray
        Reference temperature, shape (kmax, jd).
    qref : np.ndarray
        Reference QGPV, shape (kmax, nd).
    u : np.ndarray
        Reference zonal wind, shape (kmax, jd).
    """
    sjk = np.ascontiguousarray(sjk, dtype=np.float64)
    tjk = np.ascontiguousarray(tjk, dtype=np.float64)
    ckref = np.ascontiguousarray(ckref, dtype=np.float64)
    tb = np.ascontiguousarray(tb, dtype=np.float64)
    qref_over_cor = np.ascontiguousarray(qref_over_cor, dtype=np.float64)
    
    kmax, nd = qref_over_cor.shape
    jd = sjk.shape[1] + 2
    
    return _upward_sweep_core(
        int(jmax), int(kmax), int(nd), int(jb), int(jd),
        sjk, tjk, ckref, tb, qref_over_cor,
        float(a), float(om), float(dz), float(h), float(rr), float(cp)
    )
