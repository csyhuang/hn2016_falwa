"""
Numba implementation of upward_sweep.

This module provides a JIT-compiled Python implementation of the Fortran
subroutine `upward_sweep` for the upward sweep step in the direct solver
algorithm to recover u, tref, and qref.

.. versionadded:: 2.4.0
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
    
    # Initialize arrays
    pjk = np.zeros((jd - 2, kmax), dtype=np.float64)
    u = np.zeros((jd, kmax), dtype=np.float64)
    tref = np.zeros((jd, kmax), dtype=np.float64)
    qref = np.zeros((nd, kmax), dtype=np.float64)
    tg = np.zeros(kmax, dtype=np.float64)
    
    # pjk(:,1) = 0. (already initialized)
    
    # Upward sweep: Fortran k = 1, kmax-1
    for k in range(kmax - 1):  # 0 to kmax-2 (Fortran 1 to kmax-1)
        pj = np.zeros(jd - 2, dtype=np.float64)
        for i in range(jd - 2):
            pj[i] = pjk[i, k]
        
        sjj = np.zeros((jd - 2, jd - 2), dtype=np.float64)
        for i in range(jd - 2):
            for j in range(jd - 2):
                sjj[i, j] = sjk[i, j, k]
        
        tj = np.zeros(jd - 2, dtype=np.float64)
        for i in range(jd - 2):
            tj[i] = tjk[i, k]
        
        yj = np.zeros(jd - 2, dtype=np.float64)
        for i in range(jd - 2):
            yj[i] = 0.0
            for kk in range(jd - 2):
                yj[i] += sjj[i, kk] * pj[kk]
            pjk[i, k + 1] = yj[i] + tj[i]
    
    # Recover u: Fortran k = 1, kmax; j = 2, jd-1
    for k in range(kmax):
        for j in range(1, jd - 1):  # j = 2 to jd-1 in Fortran -> 1 to jd-2 in 0-indexed
            u[j, k] = pjk[j - 1, k]
    
    # Corner boundary conditions
    u[0, 0] = 0.0  # u(1,1)
    u[jd - 1, 0] = 0.0  # u(jd,1)
    # u(1,kmax) = ckref(1+jb,kmax)/(2.*pi*a)-om*a*cos(dp*float(jb))
    u[0, kmax - 1] = ckref[jb, kmax - 1] / (2.0 * pi * a) - om * a * np.cos(dp * float(jb))
    u[jd - 1, kmax - 1] = 0.0  # u(jd,kmax)
    
    # Divide by cos phi to recover Uref
    # Fortran: do jj = jb+1, nd-1
    for jj in range(jb + 1, nd):  # jb+1 to nd-1 inclusive in Fortran 1-indexed
        j = jj - jb  # Fortran: j = jj - jb
        phi0 = dp * float(jj - 1)
        for k in range(kmax):
            u[j - 1, k] = u[j - 1, k] / np.cos(phi0)  # u(j,:) in Fortran -> u[j-1,:] in 0-indexed
    
    # Extrapolate u at jd
    for k in range(kmax):
        u[jd - 1, k] = 2.0 * u[jd - 2, k] - u[jd - 3, k]
    
    # Copy qref_over_cor to qref
    for j in range(nd):
        for k in range(kmax):
            qref[j, k] = qref_over_cor[j, k]
    
    # Compute tref: Fortran k = 2, kmax-1
    for k in range(1, kmax - 1):  # k = 2 to kmax-1 in Fortran -> 1 to kmax-2 in 0-indexed
        t00 = 0.0
        zz = dz * float(k)  # dz*float(k-1) in Fortran with 1-indexed k -> dz*float(k) with 0-indexed
        tref[0, k] = t00  # tref(1,k)
        tref[1, k] = t00  # tref(2,k)
        
        # Fortran: do j = 2, jd-1
        for j in range(1, jd - 1):  # j = 2 to jd-1 in Fortran -> 1 to jd-2 in 0-indexed
            phi0 = dp * float(j + jb)  # dp*float(j-1+jb) in Fortran with 1-indexed j
            cor = 2.0 * om * np.sin(phi0)
            uz = (u[j, k + 1] - u[j, k - 1]) / (2.0 * dz)
            ty = -cor * uz * a * h * np.exp(rkappa * zz / h)
            ty = ty / rr
            tref[j + 1, k] = tref[j - 1, k] + 2.0 * ty * dp
        
        # Compute qref from qref_over_cor
        for j in range(nd):
            phi0 = dp * float(j)
            qref[j, k] = qref_over_cor[j, k] * np.sin(phi0)
        
        # Compute tg (area-weighted mean)
        tg[k] = 0.0
        wt = 0.0
        for jj in range(jb + 1, nd + 1):  # jb+1 to nd in Fortran 1-indexed
            j = jj - jb
            phi0 = dp * float(jj - 1)
            tg[k] += np.cos(phi0) * tref[j - 1, k]  # tref(j,k) in Fortran -> tref[j-1,k]
            wt += np.cos(phi0)
        tg[k] = tg[k] / wt
        
        tres = tb[k] - tg[k]
        for j in range(jd):
            tref[j, k] = tref[j, k] + tres
    
    # Boundary levels
    for j in range(jd):
        tref[j, 0] = tref[j, 1] - tb[1] + tb[0]
        tref[j, kmax - 1] = tref[j, kmax - 2] - tb[kmax - 2] + tb[kmax - 1]
    
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
        S matrix array, shape (jd-2, jd-2, kmax-1).
    tjk : np.ndarray
        T vector array, shape (jd-2, kmax-1).
    ckref : np.ndarray
        Reference Kelvin circulation, shape (nd, kmax).
    tb : np.ndarray
        Hemispheric-mean temperature, shape (kmax,).
    qref_over_cor : np.ndarray
        Reference QGPV normalized by sin(lat), shape (nd, kmax).
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
        Reference temperature, shape (jd, kmax).
    qref : np.ndarray
        Reference QGPV, shape (nd, kmax).
    u : np.ndarray
        Reference zonal wind, shape (jd, kmax).
    """
    sjk = np.ascontiguousarray(sjk, dtype=np.float64)
    tjk = np.ascontiguousarray(tjk, dtype=np.float64)
    ckref = np.ascontiguousarray(ckref, dtype=np.float64)
    tb = np.ascontiguousarray(tb, dtype=np.float64)
    qref_over_cor = np.ascontiguousarray(qref_over_cor, dtype=np.float64)
    
    nd, kmax = qref_over_cor.shape
    jd = sjk.shape[0] + 2
    
    return _upward_sweep_core(
        int(jmax), int(kmax), int(nd), int(jb), int(jd),
        sjk, tjk, ckref, tb, qref_over_cor,
        float(a), float(om), float(dz), float(h), float(rr), float(cp)
    )

